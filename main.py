import os
import time
import asyncio
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import mplfinance as mpf
from telegram import Bot
from telegram.ext import Application, CommandHandler
import google.generativeai as genai
from PIL import Image

# ==================== CONFIG ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

CRYPTO_PAIRS = ["BTC", "ETH", "SOL"]  # Deribit supports these
TIMEFRAME = "1h"  # Deribit timeframe
SCAN_INTERVAL = 300  # 5 minutes = 300 seconds
GEMINI_MODEL = "gemini-1.5-flash-latest"

# ==================== GEMINI INITIALIZATION ====================
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    print("✅ Gemini AI Client Initialized Successfully.")
except Exception as e:
    print(f"🔴 FATAL ERROR: Gemini initialization failed: {e}")
    gemini_model = None

# ==================== DERIBIT DATA FETCH ====================
async def fetch_deribit_candles(session, symbol, timeframe="60", count=720):
    """
    Fetch last 1 month (30 days) candle data from Deribit
    timeframe: 60 = 1 hour, count: 720 = 30 days * 24 hours
    """
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
                print(f"✅ Fetched {len(df)} candles for {symbol} from Deribit")
                return df
            else:
                print(f"❌ Deribit API returned error for {symbol}")
                return None
    except Exception as e:
        print(f"❌ Error fetching {symbol} candles: {e}")
        return None

async def fetch_deribit_options_chain(session, symbol):
    """Fetch complete options chain data from Deribit"""
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
                
                # Parse option chain data
                chain_data = []
                for opt in options:
                    chain_data.append({
                        'instrument': opt['instrument_name'],
                        'type': 'CALL' if 'C' in opt['instrument_name'] else 'PUT',
                        'strike': opt['instrument_name'].split('-')[2] if len(opt['instrument_name'].split('-')) > 2 else 'N/A',
                        'expiry': opt['instrument_name'].split('-')[1] if len(opt['instrument_name'].split('-')) > 1 else 'N/A',
                        'bid_price': opt.get('bid_price', 0),
                        'ask_price': opt.get('ask_price', 0),
                        'mark_price': opt.get('mark_price', 0),
                        'volume': opt.get('volume', 0),
                        'open_interest': opt.get('open_interest', 0),
                        'mid_price': opt.get('mid_price', 0)
                    })
                
                df = pd.DataFrame(chain_data)
                print(f"✅ Fetched {len(df)} options for {symbol}")
                return df
            else:
                print(f"❌ No options data for {symbol}")
                return None
    except Exception as e:
        print(f"❌ Error fetching {symbol} options chain: {e}")
        return None

async def fetch_current_price(session, symbol):
    """Fetch current spot price from Deribit"""
    url = "https://www.deribit.com/api/v2/public/ticker"
    params = {"instrument_name": f"{symbol}-PERPETUAL"}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            if data.get("result"):
                return data["result"].get("last_price", 0)
    except Exception as e:
        print(f"❌ Error fetching current price: {e}")
    return 0

# ==================== CHART GENERATION ====================
def create_chart(df, symbol):
    """Create TradingView style chart"""
    chart_file = f"chart_{symbol}_{int(time.time())}.png"
    
    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350', 
        edge='inherit', 
        wick={'up':'#26a69a', 'down':'#ef5350'}, 
        volume='in'
    )
    s = mpf.make_mpf_style(
        marketcolors=mc, 
        gridstyle='--', 
        gridcolor='#2a2e39', 
        facecolor='#1e222d', 
        edgecolor='#2a2e39', 
        figcolor='#1e222d', 
        y_on_right=True
    )
    
    try:
        mpf.plot(
            df, 
            type='candle', 
            style=s, 
            title=f"{symbol}-PERPETUAL - Last 30 Days (1H)", 
            ylabel='Price (USD)', 
            volume=True, 
            savefig=chart_file, 
            figsize=(16, 10),
            warn_too_much_data=len(df)+1
        )
        print(f"✅ Chart created: {chart_file}")
        return chart_file
    except Exception as e:
        print(f"❌ Chart error: {e}")
        return None

# ==================== OPTIONS CHAIN FORMATTING ====================
def format_options_data(options_df, current_price):
    """Format options chain data for Telegram message"""
    if options_df is None or options_df.empty:
        return "❌ No options data available"
    
    # Filter options near current price (±20%)
    options_df['strike_num'] = pd.to_numeric(options_df['strike'], errors='coerce')
    filtered = options_df[
        (options_df['strike_num'] >= current_price * 0.8) & 
        (options_df['strike_num'] <= current_price * 1.2)
    ].copy()
    
    # Sort by expiry and strike
    filtered = filtered.sort_values(['expiry', 'strike_num'])
    
    # Get top 10 by volume
    top_volume = filtered.nlargest(10, 'volume')
    
    message = f"📊 **Options Chain Summary**\n\n"
    message += f"💰 Current Price: ${current_price:,.2f}\n"
    message += f"📈 Total Options: {len(options_df)}\n\n"
    message += f"**🔥 Top 10 by Volume (Near Money):**\n\n"
    
    for idx, row in top_volume.iterrows():
        message += f"{'📗' if row['type'] == 'CALL' else '📕'} **{row['instrument']}**\n"
        message += f"   Strike: ${row['strike']} | Type: {row['type']}\n"
        message += f"   Bid/Ask: ${row['bid_price']:.4f} / ${row['ask_price']:.4f}\n"
        message += f"   Volume: {row['volume']:.2f} | OI: {row['open_interest']:.2f}\n\n"
    
    # Add summary statistics
    total_call_volume = filtered[filtered['type'] == 'CALL']['volume'].sum()
    total_put_volume = filtered[filtered['type'] == 'PUT']['volume'].sum()
    total_call_oi = filtered[filtered['type'] == 'CALL']['open_interest'].sum()
    total_put_oi = filtered[filtered['type'] == 'PUT']['open_interest'].sum()
    
    message += f"\n📊 **Statistics (±20% from spot):**\n"
    message += f"CALL Volume: {total_call_volume:,.2f} | PUT Volume: {total_put_volume:,.2f}\n"
    message += f"CALL OI: {total_call_oi:,.2f} | PUT OI: {total_put_oi:,.2f}\n"
    message += f"Put/Call Ratio: {(total_put_volume/total_call_volume if total_call_volume > 0 else 0):.2f}"
    
    return message

# ==================== GEMINI ANALYSIS ====================
def analyze_chart_with_gemini(chart_file, ohlc_data, symbol, options_summary):
    """Analyze chart with Gemini AI including options data context"""
    if not gemini_model:
        return "Gemini model is not initialized."
        
    prompt = f"""
    You are an expert crypto options trader analyzing {symbol} on Deribit.
    
    **Last 10 candles OHLC data (1H timeframe):**
    {ohlc_data.tail(10).to_string()}
    
    **Options Chain Context:**
    {options_summary}
    
    **Your task:**
    Analyze this 30-day chart for:
    1. Major Support & Resistance levels
    2. Trend analysis (bullish/bearish/sideways)
    3. Key candlestick patterns
    4. Chart patterns (triangles, H&S, channels, etc.)
    5. Volume analysis
    6. Options market sentiment based on Put/Call ratio
    7. Potential trade setups considering options data
    
    **IMPORTANT:**
    - If there's a HIGH-PROBABILITY setup, respond with: "TRADE SETUP FOUND"
    - Provide: Entry, Stop Loss, Target, Risk/Reward
    - Consider options flow in your analysis
    - If no clear setup, respond with: "NO TRADE SETUP"
    
    Keep analysis concise and actionable.
    """
    
    try:
        img = Image.open(chart_file)
        response = gemini_model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return f"Analysis failed: {str(e)}"

# ==================== TELEGRAM ALERT ====================
async def send_telegram_alert(bot, symbol, chart_file, options_message, analysis):
    """Send alert to Telegram with chart and options data"""
    try:
        # Send chart with analysis
        with open(chart_file, 'rb') as photo:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo,
                caption=f"📊 **{symbol}-PERPETUAL Analysis**\n\n{analysis[:800]}...",
                parse_mode='Markdown'
            )
        
        # Send options chain data
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=options_message,
            parse_mode='Markdown'
        )
        
        print(f"✅ Alert sent for {symbol}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# ==================== MAIN SCANNER ====================
async def scan_cryptos(bot: Bot):
    """Main scanner loop - runs every 5 minutes"""
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID, 
        text=f"🚀 **Deribit Options Bot Started!**\n\n📊 Monitoring: {', '.join(CRYPTO_PAIRS)}\n⏰ Scan Interval: {SCAN_INTERVAL//60} minutes",
        parse_mode='Markdown'
    )
    
    print(f"📊 Monitoring: {', '.join(CRYPTO_PAIRS)}")
    print(f"⏰ Scan Interval: Every {SCAN_INTERVAL//60} minutes\n")
    
    async with aiohttp.ClientSession() as session:
        while True:
            for symbol in CRYPTO_PAIRS:
                print(f"\n{'='*60}")
                print(f"🔍 Scanning {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Fetch candle data (last 30 days)
                df = await fetch_deribit_candles(session, symbol, timeframe="60", count=720)
                if df is None or df.empty:
                    print(f"⚠️ Skipping {symbol} - no candle data")
                    continue
                
                # Fetch options chain
                options_df = await fetch_deribit_options_chain(session, symbol)
                
                # Fetch current price
                current_price = await fetch_current_price(session, symbol)
                print(f"💰 Current Price: ${current_price:,.2f}")
                
                # Create chart
                chart_file = create_chart(df, symbol)
                if chart_file is None:
                    print(f"⚠️ Skipping {symbol} - chart error")
                    continue
                
                # Format options data
                options_message = format_options_data(options_df, current_price)
                
                # Analyze with Gemini
                print("🤖 Analyzing with Gemini AI...")
                analysis = analyze_chart_with_gemini(chart_file, df, symbol, options_message[:500])
                print(f"📋 Analysis Preview:\n{analysis[:300]}...")
                
                # Send alert (always send for monitoring)
                if "TRADE SETUP FOUND" in analysis.upper():
                    print(f"🎯 Trade setup detected for {symbol}!")
                else:
                    print(f"📊 Market analysis for {symbol}")
                
                await send_telegram_alert(bot, symbol, chart_file, options_message, analysis)
                
                # Cleanup
                try:
                    os.remove(chart_file)
                except OSError as e:
                    print(f"⚠️ Error removing chart: {e}")
                
                # Wait before next coin
                await asyncio.sleep(10)
            
            print(f"\n⏳ Waiting {SCAN_INTERVAL//60} minutes before next scan cycle...\n")
            await asyncio.sleep(SCAN_INTERVAL)

# ==================== TELEGRAM COMMANDS ====================
async def start(update, context):
    await update.message.reply_text(
        "🤖 **Deribit Options Chain Bot**\n\n"
        f"📊 Coins: BTC, ETH, SOL\n"
        f"⏰ Scans every 5 minutes\n"
        f"📈 30-day charts with options data\n\n"
        f"Commands:\n"
        f"/start - Bot info\n"
        f"/status - Check status",
        parse_mode='Markdown'
    )

async def status(update, context):
    await update.message.reply_text(
        f"✅ **Bot Status: Running**\n\n"
        f"📊 Monitoring: {', '.join(CRYPTO_PAIRS)}\n"
        f"⏰ Next scan in progress...",
        parse_mode='Markdown'
    )

# ==================== RUN BOT ====================
async def main():
    """Start bot and scanner"""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    
    # Start scanner in background
    application.job_queue.run_once(lambda ctx: asyncio.create_task(scan_cryptos(ctx.bot)), 0)
    
    # Start polling
    await application.run_polling()

if __name__ == "__main__":
    if gemini_model:
        asyncio.run(main())
    else:
        print("🔴 Bot cannot start - Gemini initialization failed.")
