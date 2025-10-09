import os
import time
import asyncio
from datetime import datetime
import requests
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

CRYPTO_PAIRS = [
    "SOLUSDT", "XRPUSDT", "BNBUSDT", "LTCUSDT", "AVAXUSDT",
    "LINKUSDT", "ADAUSDT", "DOGEUSDT", "TRXUSDT"
]
TIMEFRAME = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d
SCAN_INTERVAL = 60  # 60 seconds = 1 minute per crypto
GEMINI_MODEL = "gemini-1.5-flash-latest"  # flash = faster + more free requests

# ==================== BINANCE DATA FETCH ====================
def fetch_binance_data(symbol, interval="1h", limit=999):
    """Binance public API kadun OHLCV data fetch karto"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # DataFrame banvto
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Data type convert
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

# ==================== CHART GENERATION ====================
def create_chart(df, symbol, timeframe):
    """TradingView style chart banvto"""
    chart_file = f"chart_{symbol}_{int(time.time())}.png"
    
    # mplfinance style (red/green candles)
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
    
    # Chart banvto
    try:
        mpf.plot(
            df,
            type='candle',
            style=s,
            title=f"{symbol} - {timeframe.upper()}",
            ylabel='Price (USDT)',
            volume=True,
            savefig=chart_file,
            figsize=(14, 8),
            warn_too_much_data=len(df)+1
        )
        return chart_file
    except Exception as e:
        print(f"❌ Chart error: {e}")
        return None

# ==================== GEMINI ANALYSIS ====================
def analyze_chart_with_gemini(chart_file, ohlc_data, symbol):
    """Gemini 1.5 Flash use karun chart analysis karto"""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    # Analysis prompt
    prompt = f"""
    You are an expert crypto trader analyzing {symbol} chart.
    
    **Last 5 candles OHLC data:**
    {ohlc_data.tail(5).to_string()}
    
    **Your task:**
    Analyze this chart for:
    1. Support & Resistance levels
    2. Trendlines (bullish/bearish)
    3. Candlestick patterns (doji, hammer, engulfing, etc.)
    4. Chart patterns (triangle, H&S, double top/bottom, wedge, etc.)
    5. Price action signals
    
    **IMPORTANT:**
    - If there's a HIGH-PROBABILITY trade setup, respond with: "TRADE SETUP FOUND"
    - Then provide: Entry, Stop Loss, Target, Reason
    - If no clear setup, respond with: "NO TRADE SETUP"
    
    Format your response clearly.
    """
    
    try:
        img = Image.open(chart_file)
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return f"Analysis failed: {str(e)}"

# ==================== TELEGRAM ALERT ====================
async def send_telegram_alert(bot, symbol, chart_file, analysis):
    """Telegram var alert pathavto"""
    try:
        # Chart pathavto
        with open(chart_file, 'rb') as photo:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo,
                caption=f"📊 **{symbol} Analysis**\n\n{analysis}",
                parse_mode='Markdown'
            )
        print(f"✅ Alert sent for {symbol}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# ==================== MAIN SCANNER ====================
async def scan_cryptos():
    """Main scanner loop - 1 crypto per minute"""
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    print("🚀 Starting Crypto Scanner Bot...")
    print(f"📊 Monitoring: {', '.join(CRYPTO_PAIRS)}")
    print(f"⏰ Timeframe: {TIMEFRAME}")
    print(f"🔄 Scan interval: {SCAN_INTERVAL}s per crypto\n")
    
    while True:
        for symbol in CRYPTO_PAIRS:
            print(f"\n{'='*50}")
            print(f"🔍 Scanning {symbol}...")
            print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 1. Data fetch karo
            df = fetch_binance_data(symbol, TIMEFRAME, 999)
            if df is None or df.empty:
                print(f"⚠️ Skipping {symbol} - no data")
                time.sleep(SCAN_INTERVAL)
                continue
            
            print(f"✅ Fetched {len(df)} candles")
            
            # 2. Chart banvao
            chart_file = create_chart(df, symbol, TIMEFRAME)
            if chart_file is None:
                print(f"⚠️ Skipping {symbol} - chart error")
                time.sleep(SCAN_INTERVAL)
                continue
            
            print(f"✅ Chart created: {chart_file}")
            
            # 3. Gemini analysis
            print("🤖 Analyzing with Gemini...")
            analysis = analyze_chart_with_gemini(chart_file, df, symbol)
            print(f"\n📋 Analysis:\n{analysis}")
            
            # 4. Trade setup check karun alert pathavao
            if "TRADE SETUP FOUND" in analysis.upper():
                print(f"🎯 Trade setup detected for {symbol}!")
                await send_telegram_alert(bot, symbol, chart_file, analysis)
            else:
                print(f"📊 No trade setup for {symbol}")
            
            # 5. Chart file delete karo
            try:
                os.remove(chart_file)
            except:
                pass
            
            # 6. Next crypto sathi wait (60 seconds)
            print(f"⏳ Waiting {SCAN_INTERVAL}s before next scan...")
            time.sleep(SCAN_INTERVAL)

# ==================== TELEGRAM COMMANDS ====================
async def start(update, context):
    """Bot start command"""
    await update.message.reply_text(
        "🤖 **Crypto Chart Analysis Bot Active!**\n\n"
        f"📊 Monitoring: {', '.join(CRYPTO_PAIRS)}\n"
        f"⏰ Timeframe: {TIMEFRAME}\n"
        f"🔄 Scanning 1 crypto per minute\n\n"
        "You'll receive alerts when trade setups are found!",
        parse_mode='Markdown'
    )

async def status(update, context):
    """Bot status check"""
    await update.message.reply_text(
        f"✅ **Bot Status: Running**\n\n"
        f"📊 Pairs: {len(CRYPTO_PAIRS)}\n"
        f"⏰ Timeframe: {TIMEFRAME}\n"
        f"🔄 Scan interval: {SCAN_INTERVAL}s",
        parse_mode='Markdown'
    )

# ==================== RUN BOT ====================
if __name__ == "__main__":
    # Telegram bot setup
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    
    # Scanner + Bot doni run karo
    loop = asyncio.get_event_loop()
    
    # Background madhe scanner chalvao
    loop.create_task(scan_cryptos())
    
    # Telegram bot chalvao
    application.run_polling()
