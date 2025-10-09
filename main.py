import os
import time
import asyncio
from datetime import datetime
import aiohttp  # 'requests' aivaji 'aiohttp' vaprat ahe
import pandas as pd
import mplfinance as mpf
from telegram import Bot
from telegram.ext import Application, CommandHandler
import google.generativeai as genai
from PIL import Image

# ==================== CONFIG ====================
# .env file madhe ya values taka
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

# ==================== GEMINI & BOT INITIALIZATION (Ekdach karayche) ====================
# Gemini la fakt ekda configure kara
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    print("✅ Gemini AI Client Initialized Successfully.")
except Exception as e:
    print(f"🔴 FATAL ERROR: Gemini initialization failed: {e}")
    gemini_model = None

# ==================== BINANCE DATA FETCH (Sudharit) ====================
async def fetch_binance_data(session, symbol, interval="1h", limit=999):
    """Asynchronously Binance API kadun OHLCV data fetch karto"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

# ==================== CHART GENERATION (Kahi badal nahi) ====================
def create_chart(df, symbol, timeframe):
    """TradingView style chart banvto"""
    chart_file = f"chart_{symbol}_{int(time.time())}.png"
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick={'up':'#26a69a', 'down':'#ef5350'}, volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='#2a2e39', facecolor='#1e222d', edgecolor='#2a2e39', figcolor='#1e222d', y_on_right=True)
    try:
        mpf.plot(df, type='candle', style=s, title=f"{symbol} - {timeframe.upper()}", ylabel='Price (USDT)', volume=True, savefig=chart_file, figsize=(14, 8), warn_too_much_data=len(df)+1)
        return chart_file
    except Exception as e:
        print(f"❌ Chart error: {e}")
        return None

# ==================== GEMINI ANALYSIS (Sudharit) ====================
def analyze_chart_with_gemini(chart_file, ohlc_data, symbol):
    """Global Gemini model use karun chart analysis karto"""
    if not gemini_model:
        return "Gemini model is not initialized."
        
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
        response = gemini_model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return f"Analysis failed: {str(e)}"

# ==================== TELEGRAM ALERT (Kahi badal nahi) ====================
async def send_telegram_alert(bot, symbol, chart_file, analysis):
    """Telegram var alert pathavto"""
    try:
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

# ==================== MAIN SCANNER (Sudharit) ====================
async def scan_cryptos(bot: Bot):
    """Main scanner loop - as a background task"""
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🚀 Bot started! Scanner is now running in the background...")
    print(f"📊 Monitoring: {', '.join(CRYPTO_PAIRS)}")
    print(f"⏰ Timeframe: {TIMEFRAME}\n")
    
    async with aiohttp.ClientSession() as session:
        while True:
            for symbol in CRYPTO_PAIRS:
                print(f"\n{'='*50}\n🔍 Scanning {symbol} at {datetime.now().strftime('%H:%M:%S')}")
                
                df = await fetch_binance_data(session, symbol, TIMEFRAME, 999)
                if df is None or df.empty:
                    print(f"⚠️ Skipping {symbol} - no data")
                    await asyncio.sleep(SCAN_INTERVAL)  # time.sleep() aivaji asyncio.sleep()
                    continue
                
                print(f"✅ Fetched {len(df)} candles")
                chart_file = create_chart(df, symbol, TIMEFRAME)
                if chart_file is None:
                    print(f"⚠️ Skipping {symbol} - chart error")
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue
                
                print(f"✅ Chart created: {chart_file}")
                print("🤖 Analyzing with Gemini...")
                analysis = analyze_chart_with_gemini(chart_file, df, symbol)
                print(f"📋 Analysis:\n{analysis[:200]}...") # Purna analysis log na karta thodach dakhavto
                
                if "TRADE SETUP FOUND" in analysis.upper():
                    print(f"🎯 Trade setup detected for {symbol}!")
                    await send_telegram_alert(bot, symbol, chart_file, analysis)
                else:
                    print(f"📊 No trade setup for {symbol}")
                
                try:
                    os.remove(chart_file)
                except OSError as e:
                    print(f"Error removing chart file: {e}")
                
                print(f"⏳ Waiting {SCAN_INTERVAL}s before next scan...")
                await asyncio.sleep(SCAN_INTERVAL) # time.sleep() aivaji asyncio.sleep()

# ==================== TELEGRAM COMMANDS (Kahi badal nahi) ====================
async def start(update, context):
    await update.message.reply_text(
        "🤖 **Crypto Chart Analysis Bot Active!**\n\n"
        f"📊 Monitoring: {', '.join(CRYPTO_PAIRS)}\n"
        f"Scanner background madhe suru ahe.",
        parse_mode='Markdown'
    )

async def status(update, context):
    await update.message.reply_text(
        f"✅ **Bot Status: Running**",
        parse_mode='Markdown'
    )

# ==================== RUN BOT (Sudharit) ====================
async def main():
    """Starts the bot and the scanner task."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))

    # Bot la sangto ki background madhe scanner task chalav
    application.job_queue.run_once(lambda ctx: asyncio.create_task(scan_cryptos(ctx.bot)), 0)
    
    # Bot polling start karto
    await application.run_polling()

if __name__ == "__main__":
    if gemini_model:
        asyncio.run(main())
    else:
        print("Bot cannot start because Gemini initialization failed.")
