import os
import time
import asyncio
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import mplfinance as mpf
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from PIL import Image

# ==================== CONFIG ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

# Binance altcoins (Perpetual futures)
BINANCE_COINS = ["LINK", "DOGE", "XRP", "BNB", "LTC", "TRX", "ADA", "AVAX", "SOL"]

# Deribit options (BTC, ETH only)
DERIBIT_OPTIONS = ["BTC", "ETH"]

ALL_COINS = BINANCE_COINS + DERIBIT_OPTIONS
SCAN_INTERVAL = 300  # 5 minutes

print("ℹ️ Hybrid Mode: Binance altcoins + Deribit BTC/ETH options")

# ==================== BINANCE DATA FETCH ====================
async def fetch_binance_candles(session, symbol, interval="1h", limit=1000):
    """Fetch 4 months data from Binance using multiple requests"""
    all_data = []
    end_time = int(time.time() * 1000)
    
    # 4 months = ~120 days = ~2880 hourly candles
    # Binance allows max 1500 candles per request
    # We'll make 2 requests to get ~2880 candles
    
    for i in range(3):  # 3 requests to ensure 4 months coverage
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": f"{symbol}USDT",
            "interval": interval,
            "limit": 1000,
            "endTime": end_time
        }
        
        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if not data:
                    break
                
                all_data = data + all_data
                end_time = int(data[0][0]) - 1
                
                await asyncio.sleep(0.2)
        except Exception as e:
            print(f"❌ Binance error for {symbol}: {e}")
            break
    
    if all_data:
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Keep only last 4 months
        four_months_ago = datetime.now() - timedelta(days=120)
        df = df[df.index >= four_months_ago]
        
        print(f"✅ Binance: Fetched {len(df)} candles for {symbol} (Last 4 months)")
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    return None

async def fetch_binance_ticker(session, symbol):
    """Fetch 24h ticker data from Binance"""
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    params = {"symbol": f"{symbol}USDT"}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            return {
                'last_price': float(data['lastPrice']),
                'price_change_24h': float(data['priceChangePercent']),
                '24h_high': float(data['highPrice']),
                '24h_low': float(data['lowPrice']),
                'volume_24h': float(data['volume']),
                'volume_usd': float(data['quoteVolume'])
            }
    except Exception as e:
        print(f"❌ Binance ticker error: {e}")
        return None

async def fetch_binance_funding_rate(session, symbol):
    """Fetch funding rate from Binance"""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": f"{symbol}USDT", "limit": 1}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            if data:
                return float(data[0]['fundingRate']) * 100
    except Exception as e:
        print(f"❌ Funding rate error: {e}")
    return 0

async def fetch_binance_open_interest(session, symbol):
    """Fetch open interest from Binance"""
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    params = {"symbol": f"{symbol}USDT"}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            return float(data['openInterest'])
    except Exception as e:
        print(f"❌ OI error: {e}")
    return 0

async def fetch_binance_orderbook(session, symbol):
    """Fetch order book from Binance"""
    url = "https://fapi.binance.com/fapi/v1/depth"
    params = {"symbol": f"{symbol}USDT", "limit": 10}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            bids = [[float(x[0]), float(x[1])] for x in data['bids'][:5]]
            asks = [[float(x[0]), float(x[1])] for x in data['asks'][:5]]
            
            return {
                'bids': bids,
                'asks': asks,
                'best_bid': float(data['bids'][0][0]) if data['bids'] else 0,
                'best_ask': float(data['asks'][0][0]) if data['asks'] else 0,
                'spread': float(data['asks'][0][0]) - float(data['bids'][0][0]) if data['bids'] and data['asks'] else 0
            }
    except Exception as e:
        print(f"❌ Order book error: {e}")
    return None

# ==================== DERIBIT DATA FETCH ====================
async def fetch_deribit_candles(session, symbol):
    """Fetch 4 months data from Deribit"""
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    
    end_timestamp = int(time.time() * 1000)
    start_timestamp = int((datetime.now() - timedelta(days=120)).timestamp() * 1000)
    
    params = {
        "instrument_name": f"{symbol}-PERPETUAL",
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "resolution": "60"
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
                print(f"✅ Deribit: Fetched {len(df)} candles for {symbol} (Last 4 months)")
                return df
    except Exception as e:
        print(f"❌ Deribit candles error: {e}")
    return None

async def fetch_deribit_ticker(session, symbol):
    """Fetch ticker from Deribit"""
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
                    'mark_price': result.get('mark_price', 0),
                    'index_price': result.get('index_price', 0),
                    'volume_24h': result.get('stats', {}).get('volume', 0),
                    'volume_usd': result.get('stats', {}).get('volume_usd', 0),
                    'open_interest': result.get('open_interest', 0),
                    'funding_8h': result.get('funding_8h', 0),
                    '24h_high': result.get('stats', {}).get('high', 0),
                    '24h_low': result.get('stats', {}).get('low', 0),
                    'price_change_24h': result.get('stats', {}).get('price_change', 0)
                }
    except Exception as e:
        print(f"❌ Deribit ticker error: {e}")
    return None

async def fetch_deribit_options_chain(session, symbol):
    """Fetch options chain from Deribit (BTC/ETH only)"""
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    params = {"currency": symbol, "kind": "option"}
    
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
                        'volume': opt.get('volume', 0),
                        'open_interest': opt.get('open_interest', 0)
                    })
                
                df = pd.DataFrame(chain_data)
                print(f"✅ Fetched {len(df)} options for {symbol}")
                return df
    except Exception as e:
        print(f"❌ Options error: {e}")
    return None

# ==================== CHART GENERATION ====================
def create_chart(df, symbol, source="Binance"):
    """Create white background chart for 4 months data"""
    chart_file = f"chart_{symbol}_{int(time.time())}.png"
    
    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350', 
        edge='inherit', 
        wick={'up':'#26a69a', 'down':'#ef5350'}, 
        volume='in', alpha=0.9
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc, gridstyle='-', 
        gridcolor='#e0e0e0', gridaxis='both',
        facecolor='white', figcolor='white',
        edgecolor='#cccccc', rc={'font.size': 10},
        y_on_right=True
    )
    
    title = f"{symbol}USDT ({source}) | Last 4 Months (1H)"
    
    try:
        mpf.plot(
            df, type='candle', style=s, title=title,
            ylabel='Price (USDT)', volume=True, 
            savefig=chart_file, figsize=(20, 12),
            warn_too_much_data=len(df)+1, tight_layout=True
        )
        print(f"✅ Chart created: {chart_file} ({len(df)} candles)")
        return chart_file
    except Exception as e:
        print(f"❌ Chart error: {e}")
    return None

# ==================== DATA FORMATTING ====================
def format_binance_data(symbol, ticker, funding_rate, oi, order_book):
    """Format Binance perpetual data"""
    msg = f"📊 **{symbol}USDT Binance Perpetual**\n\n"
    
    if ticker:
        msg += f"💰 **Price:**\n"
        msg += f"   Last: ${ticker['last_price']:,.4f}\n"
        msg += f"   24h High: ${ticker['24h_high']:,.4f}\n"
        msg += f"   24h Low: ${ticker['24h_low']:,.4f}\n"
        msg += f"   24h Change: {ticker['price_change_24h']:.2f}%\n\n"
        
        msg += f"📈 **Volume & OI:**\n"
        msg += f"   24h Volume: {ticker['volume_24h']:,.2f} {symbol}\n"
        msg += f"   24h Volume USD: ${ticker['volume_usd']:,.0f}\n"
        msg += f"   Open Interest: {oi:,.2f} {symbol}\n\n"
    
    if funding_rate:
        funding_annual = funding_rate * 365 * 3
        msg += f"💸 **Funding Rate:**\n"
        msg += f"   Current (8h): {funding_rate:.4f}%\n"
        msg += f"   Annualized: {funding_annual:.2f}%\n\n"
    
    if order_book:
        msg += f"📖 **Order Book:**\n"
        msg += f"   Best Bid: ${order_book['best_bid']:,.4f}\n"
        msg += f"   Best Ask: ${order_book['best_ask']:,.4f}\n"
        msg += f"   Spread: ${order_book['spread']:.4f}\n\n"
        
        msg += f"   **Top 3 Bids:**\n"
        for bid in order_book['bids'][:3]:
            msg += f"   ${bid[0]:,.4f} × {bid[1]:,.2f}\n"
        
        msg += f"\n   **Top 3 Asks:**\n"
        for ask in order_book['asks'][:3]:
            msg += f"   ${ask[0]:,.4f} × {ask[1]:,.2f}\n"
    
    return msg

def format_deribit_data(symbol, ticker):
    """Format Deribit perpetual data"""
    msg = f"📊 **{symbol} Deribit Perpetual**\n\n"
    
    if ticker:
        msg += f"💰 **Price:**\n"
        msg += f"   Last: ${ticker['last_price']:,.2f}\n"
        msg += f"   Mark: ${ticker['mark_price']:,.2f}\n"
        msg += f"   Index: ${ticker['index_price']:,.2f}\n"
        msg += f"   24h High: ${ticker['24h_high']:,.2f}\n"
        msg += f"   24h Low: ${ticker['24h_low']:,.2f}\n"
        msg += f"   24h Change: {ticker['price_change_24h']:.2f}%\n\n"
        
        msg += f"📈 **Volume & OI:**\n"
        msg += f"   24h Volume: {ticker['volume_24h']:,.2f} {symbol}\n"
        msg += f"   24h Volume USD: ${ticker['volume_usd']:,.0f}\n"
        msg += f"   Open Interest: {ticker['open_interest']:,.2f}\n\n"
        
        if ticker.get('funding_8h'):
            funding = ticker['funding_8h'] * 100
            funding_annual = funding * 365 * 3
            msg += f"💸 **Funding Rate:**\n"
            msg += f"   Current (8h): {funding:.4f}%\n"
            msg += f"   Annualized: {funding_annual:.2f}%\n\n"
    
    return msg

def format_options_data(options_df, current_price, symbol):
    """Format options chain"""
    if options_df is None or options_df.empty:
        return "❌ No options data"
    
    options_df['strike_num'] = pd.to_numeric(options_df['strike'], errors='coerce')
    filtered = options_df[
        (options_df['strike_num'] >= current_price * 0.8) & 
        (options_df['strike_num'] <= current_price * 1.2)
    ].copy()
    
    top_volume = filtered.nlargest(10, 'volume')
    
    msg = f"🎯 **{symbol} Options Chain**\n\n"
    msg += f"💰 Spot: ${current_price:,.2f}\n"
    msg += f"📈 Total Options: {len(options_df)}\n\n"
    msg += f"**🔥 Top 10 by Volume (±20%):**\n\n"
    
    for idx, row in top_volume.iterrows():
        emoji = '📗' if row['type'] == 'CALL' else '📕'
        msg += f"{emoji} **{row['instrument']}**\n"
        msg += f"   Strike: ${row['strike']} | {row['type']}\n"
        msg += f"   Bid/Ask: ${row['bid_price']:.4f}/${row['ask_price']:.4f}\n"
        msg += f"   Vol: {row['volume']:.2f} | OI: {row['open_interest']:.2f}\n\n"
    
    # Stats
    call_vol = filtered[filtered['type'] == 'CALL']['volume'].sum()
    put_vol = filtered[filtered['type'] == 'PUT']['volume'].sum()
    call_oi = filtered[filtered['type'] == 'CALL']['open_interest'].sum()
    put_oi = filtered[filtered['type'] == 'PUT']['open_interest'].sum()
    pcr = (put_vol/call_vol if call_vol > 0 else 0)
    
    msg += f"\n📊 **Statistics:**\n"
    msg += f"📗 CALL Vol: {call_vol:,.2f} | OI: {call_oi:,.2f}\n"
    msg += f"📕 PUT Vol: {put_vol:,.2f} | OI: {put_oi:,.2f}\n"
    msg += f"📊 Put/Call Ratio: {pcr:.2f}"
    
    return msg

# ==================== TELEGRAM ALERT ====================
async def send_telegram_alert(bot, symbol, chart_file, market_data):
    """Send alert to Telegram"""
    try:
        with open(chart_file, 'rb') as photo:
            caption = f"📊 **{symbol} Market Data**\n_Last 4 Months Chart (1H)_"
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo,
                caption=caption,
                parse_mode='Markdown'
            )
        
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
    """Main scanner"""
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"🚀 **Crypto Scanner Started!**\n\n📊 Binance: {', '.join(BINANCE_COINS)}\n🎯 Deribit Options: {', '.join(DERIBIT_OPTIONS)}\n⏰ Scan: Every {SCAN_INTERVAL//60} mins\n📈 Chart: Last 4 Months",
            parse_mode='Markdown'
        )
    except Exception as e:
        print(f"❌ Startup error: {e}")
    
    print(f"📊 Monitoring {len(ALL_COINS)} coins (4 months data)\n")
    
    async with aiohttp.ClientSession() as session:
        while True:
            # Scan Binance altcoins
            for symbol in BINANCE_COINS:
                print(f"\n{'='*60}")
                print(f"🔍 Binance {symbol} at {datetime.now().strftime('%H:%M:%S')}")
                
                df = await fetch_binance_candles(session, symbol)
                ticker = await fetch_binance_ticker(session, symbol)
                funding = await fetch_binance_funding_rate(session, symbol)
                oi = await fetch_binance_open_interest(session, symbol)
                order_book = await fetch_binance_orderbook(session, symbol)
                
                if df is None or df.empty:
                    print(f"⚠️ Skipping {symbol}")
                    continue
                
                chart_file = create_chart(df, symbol, "Binance")
                if not chart_file:
                    continue
                
                market_data = format_binance_data(symbol, ticker, funding, oi, order_book)
                await send_telegram_alert(bot, symbol, chart_file, market_data)
                
                try:
                    os.remove(chart_file)
                except:
                    pass
                
                await asyncio.sleep(5)
            
            # Scan Deribit BTC/ETH with options
            for symbol in DERIBIT_OPTIONS:
                print(f"\n{'='*60}")
                print(f"🎯 Deribit {symbol} OPTIONS at {datetime.now().strftime('%H:%M:%S')}")
                
                df = await fetch_deribit_candles(session, symbol)
                ticker = await fetch_deribit_ticker(session, symbol)
                options_df = await fetch_deribit_options_chain(session, symbol)
                
                if df is None or df.empty:
                    continue
                
                chart_file = create_chart(df, symbol, "Deribit")
                if not chart_file:
                    continue
                
                current_price = ticker['last_price'] if ticker else 0
                perpetual_data = format_deribit_data(symbol, ticker)
                options_data = format_options_data(options_df, current_price, symbol)
                
                combined_data = perpetual_data + "\n\n" + options_data
                await send_telegram_alert(bot, symbol, chart_file, combined_data)
                
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
        "🤖 **Crypto Market Scanner**\n\n"
        f"📊 Binance: {len(BINANCE_COINS)} coins\n"
        f"🎯 Deribit: BTC/ETH options\n"
        f"⏰ Every 5 mins\n"
        f"📈 Chart: Last 4 Months\n\n/start /status",
        parse_mode='Markdown'
    )

async def status(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"✅ Running | 📊 {len(ALL_COINS)} coins | 📈 4 Months Data",
        parse_mode='Markdown'
    )

# ==================== POST INIT ====================
async def post_init(application: Application):
    print("🚀 Starting scanner...")
    asyncio.create_task(scan_cryptos(application.bot))

# ==================== RUN BOT ====================
def main():
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
    main()
