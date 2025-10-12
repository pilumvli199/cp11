import os
import time
import asyncio
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import mplfinance as mpf
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
import google.generativeai as genai
from PIL import Image

# ==================== CONFIG ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

# Top 10 Altcoins for futures data + BTC/ETH for options
ALTCOINS = ["SOL", "DOGE", "MATIC", "AVAX", "LINK", "UNI", "SHIB", "XRP", "LTC", "BCH"]
OPTIONS_COINS = ["BTC", "ETH"]  # Only these have options on Deribit
ALL_COINS = OPTIONS_COINS + ALTCOINS

SCAN_INTERVAL = 300  # 5 minutes = 300 seconds
GEMINI_MODEL = "gemini-1.5-flash-latest"

# ==================== GEMINI DISABLED ====================
# Gemini analysis disabled - only data alerts
gemini_model = None
print("ℹ️ Gemini analysis disabled - Data-only mode")

# ==================== DERIBIT DATA FETCH ====================
async def fetch_deribit_candles(session, symbol, timeframe="60", count=720):
    """Fetch last 1 month candle data from Deribit"""
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    
    end_timestamp = int(time.time() * 1000)
    start_timestamp = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
    
    params = {
        "instrument_name": f"{symbol}-PERPETUAL",
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "resolution": timeframe
    }
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            if data.get("result") and data["result"].get("status") == "ok":
                result = data["result"]
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(result['ticks'], unit='ms'),
                    'open': result['open'],
                    'high': result['high'],
                    'low': result['low'],
                    'close': result['close'],
                    'volume': result['volume']
                })
                df.set_index('timestamp', inplace=True)
                print(f"✅ Fetched {len(df)} candles for {symbol}")
                return df
            else:
                print(f"❌ Deribit API error for {symbol}")
                return None
    except Exception as e:
        print(f"❌ Error fetching {symbol} candles: {e}")
        return None

async def fetch_order_book(session, symbol):
    """Fetch order book data"""
    url = "https://www.deribit.com/api/v2/public/get_order_book"
    params = {
        "instrument_name": f"{symbol}-PERPETUAL",
        "depth": 10
    }
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            if data.get("result"):
                result = data["result"]
                return {
                    'bids': result.get('bids', [])[:5],  # Top 5 bids
                    'asks': result.get('asks', [])[:5],  # Top 5 asks
                    'best_bid': result.get('best_bid_price', 0),
                    'best_ask': result.get('best_ask_price', 0),
                    'spread': result.get('best_ask_price', 0) - result.get('best_bid_price', 0)
                }
            return None
    except Exception as e:
        print(f"❌ Error fetching order book for {symbol}: {e}")
        return None

async def fetch_funding_rate(session, symbol):
    """Fetch funding rate"""
    url = "https://www.deribit.com/api/v2/public/get_funding_rate_value"
    params = {
        "instrument_name": f"{symbol}-PERPETUAL",
        "start_timestamp": int((datetime.now() - timedelta(hours=8)).timestamp() * 1000),
        "end_timestamp": int(time.time() * 1000)
    }
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            if data.get("result"):
                # Get latest funding rate
                rates = data["result"]
                if rates:
                    latest = rates[-1]
                    return latest
            return None
    except Exception as e:
        print(f"❌ Error fetching funding rate: {e}")
        return None

async def fetch_ticker_data(session, symbol):
    """Fetch ticker data (volume, OI, price)"""
    url = "https://www.deribit.com/api/v2/public/ticker"
    params = {"instrument_name": f"{symbol}-PERPETUAL"}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            if data.get("result"):
                result = data["result"]
                return {
                    'last_price': result.get('last_price', 0),
                    'volume_24h': result.get('stats', {}).get('volume', 0),
                    'volume_usd': result.get('stats', {}).get('volume_usd', 0),
                    'open_interest': result.get('open_interest', 0),
                    'funding_8h': result.get('funding_8h', 0),
                    'mark_price': result.get('mark_price', 0),
                    'index_price': result.get('index_price', 0),
                    '24h_high': result.get('stats', {}).get('high', 0),
                    '24h_low': result.get('stats', {}).get('low', 0),
                    'price_change_24h': result.get('stats', {}).get('price_change', 0)
                }
            return None
    except Exception as e:
        print(f"❌ Error fetching ticker: {e}")
        return None

async def fetch_deribit_options_chain(session, symbol):
    """Fetch options chain (only for BTC/ETH)"""
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    params = {
        "currency": symbol,
        "kind": "option"
    }
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            if data.get("result"):
                options = data["result"]
                chain_data = []
                for opt in options:
                    chain_data.append({
                        'instrument': opt['instrument_name'],
                        'type': 'CALL' if '-C' in opt['instrument_name'] else 'PUT',
                        'strike': opt['instrument_name'].split('-')[2] if len(opt['instrument_name'].split('-')) > 2 else 'N/A',
                        'expiry': opt['instrument_name'].split('-')[1] if len(opt['instrument_name'].split('-')) > 1 else 'N/A',
                        'bid_price': opt.get('bid_price', 0),
                        'ask_price': opt.get('ask_price', 0),
                        'mark_price': opt.get('mark_price', 0),
                        'volume': opt.get('volume', 0),
                        'open_interest': opt.get('open_interest', 0)
                    })
                
                df = pd.DataFrame(chain_data)
                print(f"✅ Fetched {len(df)} options for {symbol}")
                return df
            return None
    except Exception as e:
        print(f"❌ Error fetching options: {e}")
        return None

# ==================== CHART GENERATION ====================
def create_chart(df, symbol, chart_type="perpetual"):
    """Create clean white background chart"""
    chart_file = f"chart_{symbol}_{int(time.time())}.png"
    
    # White background professional style
    mc = mpf.make_marketcolors(
        up='#26a69a', 
        down='#ef5350', 
        edge='inherit', 
        wick={'up':'#26a69a', 'down':'#ef5350'}, 
        volume='in',
        alpha=0.9
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc, 
        gridstyle='-', 
        gridcolor='#e0e0e0',
        gridaxis='both',
        facecolor='white',
        figcolor='white',
        edgecolor='#cccccc',
        rc={'font.size': 10},
        y_on_right=True
    )
    
    title = f"{symbol}-PERPETUAL | Last 30 Days (1H)"
    if chart_type == "options":
        title = f"{symbol} Options Analysis | Last 30 Days (1H)"
    
    try:
        mpf.plot(
            df, 
            type='candle', 
            style=s, 
            title=title,
            ylabel='Price (USD)', 
            volume=True, 
            savefig=chart_file, 
            figsize=(16, 10),
            warn_too_much_data=len(df)+1,
            tight_layout=True
        )
        print(f"✅ Chart created: {chart_file}")
        return chart_file
    except Exception as e:
        print(f"❌ Chart error: {e}")
        return None

# ==================== DATA FORMATTING ====================
def format_altcoin_data(symbol, ticker, order_book, funding):
    """Format altcoin perpetual futures data"""
    msg = f"📊 **{symbol}-PERPETUAL Market Data**\n\n"
    
    # Price Info
    if ticker:
        msg += f"💰 **Price Information:**\n"
        msg += f"   Last Price: ${ticker['last_price']:,.2f}\n"
        msg += f"   Mark Price: ${ticker['mark_price']:,.2f}\n"
        msg += f"   Index Price: ${ticker['index_price']:,.2f}\n"
        msg += f"   24h High: ${ticker['24h_high']:,.2f}\n"
        msg += f"   24h Low: ${ticker['24h_low']:,.2f}\n"
        msg += f"   24h Change: {ticker['price_change_24h']:.2f}%\n\n"
    
    # Volume & OI
    if ticker:
        msg += f"📈 **Volume & Open Interest:**\n"
        msg += f"   24h Volume: {ticker['volume_24h']:,.2f} {symbol}\n"
        msg += f"   24h Volume USD: ${ticker['volume_usd']:,.0f}\n"
        msg += f"   Open Interest: {ticker['open_interest']:,.2f} {symbol}\n\n"
    
    # Funding Rate
    if ticker and ticker.get('funding_8h'):
        funding_rate = ticker['funding_8h'] * 100
        funding_annual = funding_rate * 365 * 3  # 8h intervals
        msg += f"💸 **Funding Rate:**\n"
        msg += f"   Current (8h): {funding_rate:.4f}%\n"
        msg += f"   Annualized: {funding_annual:.2f}%\n\n"
    
    # Order Book
    if order_book:
        msg += f"📖 **Order Book (Top 5):**\n"
        msg += f"   Best Bid: ${order_book['best_bid']:,.2f}\n"
        msg += f"   Best Ask: ${order_book['best_ask']:,.2f}\n"
        msg += f"   Spread: ${order_book['spread']:.2f}\n\n"
        
        msg += f"   **Bids:**\n"
        for bid in order_book['bids'][:3]:
            msg += f"   ${bid[0]:,.2f} × {bid[1]:.2f}\n"
        
        msg += f"\n   **Asks:**\n"
        for ask in order_book['asks'][:3]:
            msg += f"   ${ask[0]:,.2f} × {ask[1]:.2f}\n"
    
    return msg

def format_options_data(options_df, current_price, symbol):
    """Format options chain data for BTC/ETH"""
    if options_df is None or options_df.empty:
        return "❌ No options data available"
    
    options_df['strike_num'] = pd.to_numeric(options_df['strike'], errors='coerce')
    filtered = options_df[
        (options_df['strike_num'] >= current_price * 0.8) & 
        (options_df['strike_num'] <= current_price * 1.2)
    ].copy()
    
    top_volume = filtered.nlargest(10, 'volume')
    
    msg = f"📊 **{symbol} Options Chain Summary**\n\n"
    msg += f"💰 Current Price: ${current_price:,.2f}\n"
    msg += f"📈 Total Options: {len(options_df)}\n\n"
    msg += f"**🔥 Top 10 by Volume (±20%):**\n\n"
    
    for idx, row in top_volume.iterrows():
        emoji = '📗' if row['type'] == 'CALL' else '📕'
        msg += f"{emoji} **{row['instrument']}**\n"
        msg += f"   Strike: ${row['strike']} | {row['type']}\n"
        msg += f"   Bid/Ask: ${row['bid_price']:.4f}/${row['ask_price']:.4f}\n"
        msg += f"   Vol: {row['volume']:.2f} | OI: {row['open_interest']:.2f}\n\n"
    
    # Statistics
    total_call_vol = filtered[filtered['type'] == 'CALL']['volume'].sum()
    total_put_vol = filtered[filtered['type'] == 'PUT']['volume'].sum()
    total_call_oi = filtered[filtered['type'] == 'CALL']['open_interest'].sum()
    total_put_oi = filtered[filtered['type'] == 'PUT']['open_interest'].sum()
    
    pcr = (total_put_vol/total_call_vol if total_call_vol > 0 else 0)
    
    msg += f"\n📊 **Statistics:**\n"
    msg += f"📗 CALL Vol: {total_call_vol:,.2f} | OI: {total_call_oi:,.2f}\n"
    msg += f"📕 PUT Vol: {total_put_vol:,.2f} | OI: {total_put_oi:,.2f}\n"
    msg += f"📊 Put/Call Ratio: {pcr:.2f}"
    
    return msg

# ==================== GEMINI ANALYSIS ====================
def analyze_chart_with_gemini(chart_file, ohlc_data, symbol, market_data):
    """Analyze chart with Gemini AI"""
    if not gemini_model:
        return "Gemini model not initialized."
        
    prompt = f"""
Analyze {symbol} chart and market data.

**Last 10 Candles (1H):**
{ohlc_data.tail(10).to_string()}

**Market Data:**
{market_data[:800]}

**Provide:**
1. Key Support & Resistance
2. Trend Analysis
3. Patterns detected
4. Volume analysis
5. Trade setup (if found)

If HIGH-PROBABILITY setup: "TRADE SETUP FOUND" with Entry, SL, Target
If no setup: "NO TRADE SETUP" with brief analysis

Keep concise.
"""
    
    try:
        img = Image.open(chart_file)
        response = gemini_model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return f"Analysis failed: {str(e)}"

# ==================== TELEGRAM ALERT ====================
async def send_telegram_alert(bot, symbol, chart_file, market_data, analysis):
    """Send complete alert to Telegram"""
    try:
        # Send chart with analysis
        with open(chart_file, 'rb') as photo:
            caption = f"📊 **{symbol} Analysis**\n\n{analysis[:900]}"
            if len(analysis) > 900:
                caption += "..."
            
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo,
                caption=caption,
                parse_mode='Markdown'
            )
        
        # Send market data
        if len(market_data) > 4000:
            chunks = [market_data[i:i+4000] for i in range(0, len(market_data), 4000)]
            for chunk in chunks:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=chunk,
                    parse_mode='Markdown'
                )
        else:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=market_data,
                parse_mode='Markdown'
            )
        
        print(f"✅ Alert sent for {symbol}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# ==================== MAIN SCANNER ====================
async def scan_cryptos(bot: Bot):
    """Main scanner - scans all coins every 5 minutes"""
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, 
            text=f"🚀 **Deribit Market Scanner Started!**\n\n📊 Altcoins (Perpetuals): {', '.join(ALTCOINS)}\n🎯 Options: {', '.join(OPTIONS_COINS)}\n⏰ Scan: Every {SCAN_INTERVAL//60} mins",
            parse_mode='Markdown'
        )
    except Exception as e:
        print(f"❌ Startup message error: {e}")
    
    print(f"📊 Monitoring {len(ALL_COINS)} coins")
    print(f"⏰ Scan Interval: Every {SCAN_INTERVAL//60} minutes\n")
    
    async with aiohttp.ClientSession() as session:
        while True:
            # Scan Altcoins (Perpetuals)
            for symbol in ALTCOINS:
                print(f"\n{'='*60}")
                print(f"🔍 Scanning {symbol} at {datetime.now().strftime('%H:%M:%S')}")
                
                # Fetch all data
                df = await fetch_deribit_candles(session, symbol)
                ticker = await fetch_ticker_data(session, symbol)
                order_book = await fetch_order_book(session, symbol)
                funding = await fetch_funding_rate(session, symbol)
                
                if df is None or df.empty:
                    print(f"⚠️ Skipping {symbol}")
                    continue
                
                # Create chart
                chart_file = create_chart(df, symbol, "perpetual")
                if not chart_file:
                    continue
                
                # Format data
                market_data = format_altcoin_data(symbol, ticker, order_book, funding)
                
                # AI Analysis
                print("🤖 Analyzing...")
                analysis = analyze_chart_with_gemini(chart_file, df, symbol, market_data[:500])
                
                # Send alert
                await send_telegram_alert(bot, symbol, chart_file, market_data, analysis)
                
                # Cleanup
                try:
                    os.remove(chart_file)
                except:
                    pass
                
                await asyncio.sleep(5)
            
            # Scan BTC/ETH (Options + Perpetuals)
            for symbol in OPTIONS_COINS:
                print(f"\n{'='*60}")
                print(f"🎯 Scanning {symbol} OPTIONS at {datetime.now().strftime('%H:%M:%S')}")
                
                # Fetch data
                df = await fetch_deribit_candles(session, symbol)
                ticker = await fetch_ticker_data(session, symbol)
                order_book = await fetch_order_book(session, symbol)
                options_df = await fetch_deribit_options_chain(session, symbol)
                
                if df is None or df.empty:
                    continue
                
                # Create chart
                chart_file = create_chart(df, symbol, "options")
                if not chart_file:
                    continue
                
                # Format data
                current_price = ticker['last_price'] if ticker else 0
                perpetual_data = format_altcoin_data(symbol, ticker, order_book, None)
                options_data = format_options_data(options_df, current_price, symbol)
                
                combined_data = perpetual_data + "\n\n" + options_data
                
                # AI Analysis
                print("🤖 Analyzing...")
                analysis = analyze_chart_with_gemini(chart_file, df, symbol, combined_data[:800])
                
                # Send alert
                await send_telegram_alert(bot, symbol, chart_file, combined_data, analysis)
                
                # Cleanup
                try:
                    os.remove(chart_file)
                except:
                    pass
                
                await asyncio.sleep(5)
            
            print(f"\n⏳ Waiting {SCAN_INTERVAL//60} minutes...\n")
            await asyncio.sleep(SCAN_INTERVAL)

# ==================== TELEGRAM COMMANDS ====================
async def start(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **Deribit Market Scanner**\n\n"
        f"📊 Altcoins: {len(ALTCOINS)} (Perpetuals)\n"
        f"🎯 Options: BTC, ETH\n"
        f"⏰ Scans: Every 5 mins\n\n"
        f"Commands:\n/start /status",
        parse_mode='Markdown'
    )

async def status(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"✅ Bot Running\n📊 Monitoring: {len(ALL_COINS)} coins",
        parse_mode='Markdown'
    )

# ==================== POST INIT ====================
async def post_init(application: Application):
    """Start scanner after bot init"""
    print("🚀 Starting scanner...")
    asyncio.create_task(scan_cryptos(application.bot))

# ==================== RUN BOT ====================
def main():
    """Start bot"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.post_init = post_init
    
    print("🤖 Starting bot...")
    application.run_polling()

if __name__ == "__main__":
    if gemini_model:
        main()
    else:
        print("🔴 Gemini initialization failed.")
