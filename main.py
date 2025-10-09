#!/usr/bin/env python3
# main.py - FULL Hybrid Gemini Bot with BUILT-IN Diagnostic Check
# Scan interval set to 30 minutes (1800 seconds)

import os, json, asyncio, traceback, time, sys
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from tempfile import NamedTemporaryFile
import re
from typing import Dict, Any

# plotting (server-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf

# Gemini AI client
import google.generativeai as genai
from PIL import Image

# Redis (non-TLS)
import redis

# Load environment variables
load_dotenv()

# ---------------- CONFIG (Gemini Hybrid Mode & 30 Min Scan) ----------------
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
VISION_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-pro-vision")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-pro")
POLL_INTERVAL = max(60, int(os.getenv("POLL_INTERVAL", 1800)))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0))
BASE_URL = "https://api.binance.com"
OPTIONS_BASE_URL = "https://eapi.binance.com"
CANDLE_URL = BASE_URL + "/api/v3/klines?symbol={symbol}&interval={tf}&limit={limit}"
OPTIONS_TICKER_URL = OPTIONS_BASE_URL + "/eapi/v1/ticker"
CANDLE_LIMITS = {"1h": 999, "4h": 999, "1d": 999}

for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    if var in os.environ: del os.environ[var]

# ---------------- DIAGNOSTIC CHECK FUNCTION ----------------
def run_diagnostic_check():
    """Checks library version and model availability before starting the bot."""
    print("--- Running Pre-flight Diagnostic Check ---")
    
    # 1. Check Library Version
    print(f"✅ Python Library Version: {genai.__version__}")

    # 2. Check API Key
    if not GEMINI_API_KEY:
        print("🔴 FATAL ERROR: GEMINI_API_KEY not found in your .env file!")
        sys.exit(1) # Exit the script
    print("✅ Gemini API Key found.")

    # 3. Check Model Availability
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        print("\nChecking for required models...")
        vision_model_found = f'models/{GEMINI_VISION_MODEL}' in available_models
        text_model_found = f'models/{GEMINI_TEXT_MODEL}' in available_models

        if vision_model_found:
            print(f"✅ Vision Model ({GEMINI_VISION_MODEL}) is available.")
        else:
            print(f"🔴 FATAL ERROR: Vision Model ('{GEMINI_VISION_MODEL}') is NOT AVAILABLE for your API key.")
            print("   ACTION: Please update the library by running: pip install --upgrade google-generativeai")
            sys.exit(1)

        if text_model_found:
            print(f"✅ Text Model ({GEMINI_TEXT_MODEL}) is available.")
        else:
            print(f"🔴 FATAL ERROR: Text Model ('{GEMINI_TEXT_MODEL}') is NOT AVAILABLE for your API key.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n🔴 An unexpected error occurred during check: {e}")
        print("   Please check your API key and internet connection.")
        sys.exit(1)

    print("\n--- Diagnostic Check Passed. Starting Bot... ---\n")
    return True

# === Main Bot Code Starts Here (No changes below unless specified) ===

# --- Redis Init/Helpers ---
REDIS = None
def init_redis_plain():
    global REDIS
    # (No changes to this function)
    host = os.getenv("REDIS_HOST"); port = int(os.getenv("REDIS_PORT")) if os.getenv("REDIS_PORT") else None
    user = os.getenv("REDIS_USER") or None; password = os.getenv("REDIS_PASSWORD") or None
    url = os.getenv("REDIS_URL")
    if host and port:
        try: REDIS = redis.Redis(host=host, port=port, username=user, password=password, decode_responses=True); REDIS.ping(); return
        except Exception: REDIS = None
    if url and not REDIS:
        try: REDIS = redis.Redis.from_url(url, decode_responses=True); REDIS.ping(); return
        except Exception: REDIS = None

def store_signal(symbol, signal):
    try:
        if REDIS: REDIS.hset("signals:advanced", symbol, json.dumps(signal))
    except Exception as e: print("Redis error:", e)

# ---------------- Options Chain Analysis ----------------
async def get_options_data_for_symbol(session, base_symbol) -> Dict[str, Any]:
    # (No changes to this function)
    data = {"options_sentiment": "Neutral", "oi_summary": "N/A"}
    current_time_ms = int(time.time() * 1000)
    all_tickers = await fetch_json(session, OPTIONS_TICKER_URL)
    if not all_tickers or not isinstance(all_tickers, list): return data
    target_underlying = base_symbol[:-4]
    target_tickers = [t for t in all_tickers if t.get('underlying') == target_underlying]
    if not target_tickers: return data
    all_expiries = sorted(list(set(t.get('expiryDate', 0) for t in target_tickers)))
    next_expiry_date_ms = next((exp for exp in all_expiries if exp > current_time_ms), None)
    if not next_expiry_date_ms: return data
    next_expiry_contracts = [t for t in target_tickers if t.get('expiryDate') == next_expiry_date_ms]
    if not next_expiry_contracts: return data
    total_call_oi = sum(float(t.get('openInterest', 0)) for t in next_expiry_contracts if t.get('optionType') == 'CALL')
    total_put_oi = sum(float(t.get('openInterest', 0)) for t in next_expiry_contracts if t.get('optionType') == 'PUT')
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
    if pcr == 0: sentiment = "No Open Interest."
    elif pcr < 0.7: sentiment = "Strong Call Bias (Bullish)"
    elif pcr > 1.2: sentiment = "Strong Put Bias (Bearish)"
    else: sentiment = f"Neutral Sentiment (PCR: {pcr:.2f})"
    data["options_sentiment"] = sentiment
    data["oi_summary"] = f"NextExpiry: {datetime.fromtimestamp(next_expiry_date_ms/1000).strftime('%Y-%m-%d')}, PCR: {pcr:.2f}"
    return data

# ---------------- Utility Functions ----------------
def atr(highs, lows, closes, period=14):
    high = pd.Series(highs); low = pd.Series(lows); close = pd.Series(closes).shift(1)
    tr1 = high - low; tr2 = abs(high - close); tr3 = abs(low - close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False, min_periods=period).mean().iloc[-1]

def fmt_price(p): return f"{p:.4f}" if abs(p) < 1 else f"{p:.2f}"

def parse_ai_signal(ai_output):
    # (No changes to this function)
    try:
        match = re.search(r"ACTION:\s*(BUY|SELL|HOLD|NONE)\s*-\s*ENTRY:([\d.]+)\s*-\s*SL:([\d.]+)\s*-\s*TP:([\d.]+)\s*-\s*TP2:([\d.]+)\s*-\s*REASON:(.*?)\s*-\s*CONF:(\d+\.?\d*)%", ai_output, re.IGNORECASE | re.DOTALL)
        if match:
            return {"side": match.group(1).upper().strip(), "entry": float(match.group(2).strip()), "sl": float(match.group(3).strip()), "tp": float(match.group(4).strip()), "tp2": float(match.group(5).strip()), "reason": match.group(6).strip(), "confidence": float(match.group(7).strip())}
    except Exception: pass
    return {"side": "none", "confidence": 0.0, "reason": "AI format error."}

def plot_signal_chart(symbol, candles, signal):
    # (No changes to this function)
    df = pd.DataFrame(candles, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
    df = df.set_index(pd.DatetimeIndex(df['OpenTime']))[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df_plot = df.iloc[-300:]
    apds = [mpf.make_addplot(df_plot['SMA200'], color='#FFA500')]
    hlines, colors, ls = [], [], []
    if signal["side"] in ["BUY", "SELL"]:
        hlines.extend([signal.get('entry', 0), signal.get('sl', 0), signal.get('tp', 0)])
        colors.extend(['blue', 'red', 'green'])
        ls.extend(['-', '--', '--'])
    s = mpf.make_mpf_style(base_mpf_style='yahoo', facecolor='#ffffff')
    fig, _ = mpf.plot(df_plot, type='candle', style=s, title=f"{symbol} 1H Signal | {signal.get('side','?')} | Conf {signal.get('confidence',0):.1f}%", ylabel='Price', addplot=apds, figscale=1.5, returnfig=True, hlines=dict(hlines=hlines, colors=colors, linestyle=ls, linewidths=1.5))
    tmp = NamedTemporaryFile(delete=False, suffix=f"_{symbol}.png"); fig.savefig(tmp.name, bbox_inches='tight'); plt.close(fig)
    return tmp.name

# ---------------- Gemini AI Analysis Function ----------------
async def analyze_with_gemini(symbol, data, chart_path=None):
    model_to_use = GEMINI_VISION_MODEL if symbol in VISION_SYMBOLS and chart_path else GEMINI_TEXT_MODEL
    df_1h = pd.DataFrame(data["1h"], columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df_1h = df_1h[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    current_price = df_1h['Close'].iloc[-1]
    current_atr = atr(df_1h['High'].to_numpy(), df_1h['Low'].to_numpy(), df_1h['Close'].to_numpy())
    last_10_candles_raw = df_1h.tail(10).to_string()
    text_prompt = (f"Analyze the provided chart image for {symbol} to find a trade setup. Use the text data for confirmation. Focus on chart patterns, candlestick patterns, and key levels. Combine this with the option chain sentiment to decide. Options Data: {data.get('options', {}).get('oi_summary', 'N/A')}. Current Price: {current_price:.2f}. Last 10 Candles:\n{last_10_candles_raw}\n\nStrictly reply in the format: ACTION:TYPE - ENTRY:x - SL:y - TP:z - TP2:w - REASON:.. - CONF:n%")
    prompt_parts = [text_prompt]
    if model_to_use == GEMINI_VISION_MODEL:
        try: prompt_parts.append(Image.open(chart_path))
        except Exception as e: print(f"Chart image error: {e}"); model_to_use = GEMINI_TEXT_MODEL
    try:
        model = genai.GenerativeModel(model_to_use)
        response = await asyncio.get_running_loop().run_in_executor(None, lambda: model.generate_content(prompt_parts))
        ai_output = response.text if response.parts else "ACTION:NONE - REASON:Response blocked."
        signal = parse_ai_signal(ai_output)
        signal.update({'model': model_to_use})
        return signal
    except Exception as e:
        print(f"Gemini analysis error for {symbol}: {e}")
        return {"side":"none", "confidence":0, "reason":f"AI_CALL_ERROR", "model": model_to_use}

# ---------------- Fetch & Telegram ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=20) as r: return await r.json() if r.status == 200 else None
    except Exception: return None

async def send_text(session, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try: await session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})
    except Exception as e: print("send_text error:", e)

async def send_photo(session, caption, path):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(path, "rb") as f:
        data = aiohttp.FormData({'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'photo': f, 'parse_mode': 'Markdown'})
        try: await session.post(url, data=data)
        except Exception as e: print("send_photo error:", e)

# ---------------- Main loop ----------------
async def advanced_options_loop():
    init_redis_plain()
    async with aiohttp.ClientSession() as session:
        startup = f"🤖 Bot Started ({GEMINI_VISION_MODEL}) • Scan Interval: {POLL_INTERVAL//60} min"
        await send_text(session, startup)
        it = 0
        while True:
            it += 1; print(f"\nITER {it} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = time.time()
            all_data = {}
            for sym in SYMBOLS:
                tasks = {"1h": fetch_json(session, CANDLE_URL.format(symbol=sym, tf="1h", limit=CANDLE_LIMITS["1h"])), "options": get_options_data_for_symbol(session, sym)}
                results = await asyncio.gather(*tasks.values()); all_data[sym] = dict(zip(tasks.keys(), results))
            for sym in SYMBOLS:
                chart_path = None
                try:
                    data = all_data.get(sym, {})
                    if not data.get("1h"): print(f"{sym}: Missing 1h data"); continue
                    chart_path = plot_signal_chart(sym, data["1h"], {"side":"Analyzing..."})
                    final_signal = await analyze_with_gemini(sym, data, chart_path)
                    if final_signal["side"] in ["BUY", "SELL"] and final_signal["confidence"] >= SIGNAL_CONF_THRESHOLD:
                        store_signal(sym, final_signal)
                        msg = (f"**🔥 AI Alert ({final_signal['confidence']:.1f}%)**\n\n**Asset:** {sym}\n**Action:** {final_signal['side']}\n**Entry:** `{fmt_price(final_signal['entry'])}`\n**SL:** `{fmt_price(final_signal['sl'])}` | **TP1:** `{fmt_price(final_signal['tp'])}`\n\n**Logic:** _{final_signal['reason']}_")
                        final_chart_path = plot_signal_chart(sym, data["1h"], final_signal)
                        await send_photo(session, msg, final_chart_path)
                        print(f"⚡ Alert Sent for {sym}: {final_signal['side']} @ {final_signal['entry']:.2f}")
                        if os.path.exists(final_chart_path): os.remove(final_chart_path)
                    else:
                        print(f"{sym}: No trade found. (AI: {final_signal['side']}, Conf: {final_signal['confidence']:.1f}%)")
                except Exception as e: print(f"Error processing {sym}: {e}"); traceback.print_exc()
                finally:
                    if chart_path and os.path.exists(chart_path): os.remove(chart_path)
                    await asyncio.sleep(1)
            print(f"Processing time: {time.time() - start_time:.2f}s")
            time_to_wait = POLL_INTERVAL - (time.time() - start_time)
            if time_to_wait > 0: print(f"Sleeping for {time_to_wait:.0f}s..."); await asyncio.sleep(time_to_wait)

if __name__ == "__main__":
    # The diagnostic check runs here first.
    # If it fails, the script will exit.
    # If it passes, it will start the bot.
    run_diagnostic_check()
    
    try:
        asyncio.run(advanced_options_loop())
    except KeyboardInterrupt:
        print("\nStopped by user.")
