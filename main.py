#!/usr/bin/env python3
# main.py - FULL Hybrid Gemini 1.5 Flash (Vision + Text) Multi-Timeframe Bot
# Scan interval set to 30 minutes (1800 seconds)

import os, json, asyncio, traceback, time
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
# Symbols: BTC and ETH for analysis
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Symbols that require Gemini Vision (image analysis).
VISION_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
# Using latest models for best performance
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-flash-latest")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-1.0-pro")

# *** CHANGED: SET SCAN INTERVAL TO 30 MINUTES (1800 seconds) ***
POLL_INTERVAL = max(60, int(os.getenv("POLL_INTERVAL", 1800)))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0))

# API Endpoints
BASE_URL = "https://api.binance.com"
OPTIONS_BASE_URL = "https://eapi.binance.com"
CANDLE_URL = BASE_URL + "/api/v3/klines?symbol={symbol}&interval={tf}&limit={limit}"
TICKER_24H_URL = BASE_URL + "/api/v3/ticker/24hr?symbol={symbol}"
DEPTH_URL = BASE_URL + "/api/v3/depth?symbol={symbol}&limit=50"
AGGT_URL = BASE_URL + "/api/v3/aggTrades?symbol={symbol}&limit=100"
OPTIONS_TICKER_URL = OPTIONS_BASE_URL + "/eapi/v1/ticker"
CANDLE_LIMITS = {"1h": 999, "4h": 999, "1d": 999} # Fetches 999 candles

# --- STABLE PROXY FIX ---
for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    if var in os.environ:
        del os.environ[var]

# === Gemini Client Initialization ===
client = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        client = True
        print("✅ Gemini AI Client Initialized Successfully.")
    except Exception as e:
        print(f"🔴 Error initializing Gemini client: {e}")
else:
    print("🔴 GEMINI_API_KEY not found in environment variables.")
# -------------------------------------------------------------------------

# --- Redis Init/Helpers ---
REDIS = None
def init_redis_plain():
    global REDIS
    host = os.getenv("REDIS_HOST")
    port = int(os.getenv("REDIS_PORT")) if os.getenv("REDIS_PORT") else None
    user = os.getenv("REDIS_USER") or None
    password = os.getenv("REDIS_PASSWORD") or None
    url = os.getenv("REDIS_URL")

    if host and port:
        try:
            REDIS = redis.Redis(host=host, port=port, username=user, password=password, decode_responses=True)
            REDIS.ping()
            return
        except Exception: REDIS = None
    if url and not REDIS:
        try:
            REDIS = redis.Redis.from_url(url, decode_responses=True)
            REDIS.ping()
            return
        except Exception: REDIS = None

def safe_call(fn, *args, **kwargs):
    try:
        if REDIS: return fn(*args, **kwargs)
    except Exception as e: print("Redis error:", e)
    return None

def safe_hset(h, field, val):
    return safe_call(REDIS.hset, h, field, val) if REDIS else None

def store_signal(symbol, signal): safe_hset("signals:advanced", symbol, json.dumps(signal))

# ---------------- Options Chain Analysis ----------------
async def get_options_data_for_symbol(session, base_symbol) -> Dict[str, Any]:
    data = {"options_sentiment": "Neutral", "oi_summary": "N/A", "near_term_symbol": None}
    current_time_ms = int(time.time() * 1000)

    all_tickers = await fetch_json(session, OPTIONS_TICKER_URL)
    if not all_tickers or not isinstance(all_tickers, list): return data

    target_underlying = base_symbol[:-4]
    target_tickers = [t for t in all_tickers if t.get('underlying') == target_underlying]
    if not target_tickers: return data

    all_expiries = sorted(list(set(t.get('expiryDate', 0) for t in target_tickers)))
    next_expiry_date_ms = next(
        (exp for exp in all_expiries if exp > current_time_ms),
        None
    )
    if not next_expiry_date_ms:
        data['oi_summary'] = f"Warning: No future option expiries found for {target_underlying}."
        return data

    next_expiry_contracts = [t for t in target_tickers if t.get('expiryDate') == next_expiry_date_ms]
    if not next_expiry_contracts: return data

    total_call_oi = 0; total_put_oi = 0; all_ivs = []

    for ticker in next_expiry_contracts:
        try:
            oi = float(ticker.get('openInterest', 0))
            iv = float(ticker.get('impliedVolatility', 0))
            option_type = ticker.get('optionType')

            if option_type == 'CALL': total_call_oi += oi
            elif option_type == 'PUT': total_put_oi += oi
            if iv > 0: all_ivs.append(iv)
        except Exception: continue

    total_oi = total_call_oi + total_put_oi
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
    avg_iv = np.mean(all_ivs) if all_ivs else 0

    if total_oi == 0: sentiment = "No Open Interest."
    elif pcr < 0.7: sentiment = "Strong Call Bias (Bullish sentiment) - PCR < 0.7."
    elif pcr > 1.2: sentiment = "Strong Put Bias (Bearish/Hedging sentiment) - PCR > 1.2."
    else: sentiment = f"Neutral to Moderate Sentiment (PCR: {pcr:.2f})."

    data["options_sentiment"] = sentiment
    data["oi_summary"] = (
        f"Next Expiry: {datetime.fromtimestamp(next_expiry_date_ms/1000).strftime('%Y-%m-%d')}, "
        f"Total OI: {total_oi:.2f}, Call OI: {total_call_oi:.2f}, Put OI: {total_put_oi:.2f}, "
        f"PCR: {pcr:.2f}, Avg IV: {avg_iv * 100:.2f}%."
    )
    data["near_term_symbol"] = next_expiry_contracts[0]['symbol'] if next_expiry_contracts else 'N/A'
    return data

# ---------------- Utility Functions ----------------

def rsi(prices, period=14):
    delta = pd.Series(prices).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def atr(highs, lows, closes, period=14):
    high = pd.Series(highs); low = pd.Series(lows); close = pd.Series(closes).shift(1)
    tr1 = high - low; tr2 = abs(high - close); tr3 = abs(low - close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False, min_periods=period).mean()
    return atr.iloc[-1]

def fmt_price(p):
    return f"{p:.4f}" if abs(p) < 1 else f"{p:.2f}"

def parse_ai_signal(ai_output, symbol, current_atr):
    signal = {"side": "none", "confidence": 0.0, "reason": "AI did not adhere to the format."}
    try:
        match = re.search(
             r"ACTION:\s*(BUY|SELL|HOLD|NONE)\s*-\s*ENTRY:([\d.]+)\s*-\s*SL:([\d.]+)\s*-\s*TP:([\d.]+)\s*-\s*TP2:([\d.]+)\s*-\s*REASON:(.*?)\s*-\s*CONF:(\d+\.?\d*)%",
             ai_output, re.IGNORECASE | re.DOTALL
        )
        if not match:
            action_match = re.search(r"ACTION:\s*([A-Z]+)", ai_output, re.IGNORECASE)
            conf_match = re.search(r"CONF:(\d+\.?\d*)%", ai_output, re.IGNORECASE)
            reason_match = re.search(r"REASON:(.*)", ai_output, re.IGNORECASE | re.DOTALL)
            fallback_action = action_match.group(1).upper().strip() if action_match else 'NONE'
            fallback_conf = float(conf_match.group(1).strip()) if conf_match else 0.0
            fallback_reason = reason_match.group(1).strip() if reason_match else 'Format error.'
            signal.update({"side": fallback_action, "confidence": fallback_conf, "reason": fallback_reason})
            return signal

        action = match.group(1).upper().strip()
        signal.update({
            "side": action, "entry": float(match.group(2).strip()), "sl": float(match.group(3).strip()),
            "tp": float(match.group(4).strip()), "tp2": float(match.group(5).strip()),
            "reason": match.group(6).strip(), "confidence": float(match.group(7).strip())
        })
    except Exception as e:
        signal["reason"] += f" | Parsing Exception: {e}"
    return signal

def plot_signal_chart(symbol, candles, signal):
    df_candles = pd.DataFrame(candles, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df_candles['OpenTime'] = pd.to_datetime(df_candles['OpenTime'], unit='ms')
    df_candles = df_candles.set_index(pd.DatetimeIndex(df_candles['OpenTime']))
    df_candles = df_candles[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df_candles['SMA200'] = df_candles['Close'].rolling(window=200).mean()
    df_plot = df_candles.iloc[-300:]
    apds = [mpf.make_addplot(df_plot['SMA200'], color='#FFA500', panel=0, width=1.0)]
    hlines, colors, linestyles = [], [], []
    if signal["side"] in ["BUY", "SELL"]:
        if signal.get('entry', 0) > 0: hlines.append(signal["entry"]); colors.append('blue'); linestyles.append('-')
        if signal.get('sl', 0) > 0: hlines.append(signal["sl"]); colors.append('red'); linestyles.append('--')
        if signal.get('tp', 0) > 0: hlines.append(signal["tp"]); colors.append('green'); linestyles.append('--')
    s = mpf.make_mpf_style(base_mpf_style='yahoo', facecolor='#ffffff')
    fig, axlist = mpf.plot(
        df_plot, type='candle', style=s,
        title=f"{symbol} 1H Signal | {signal.get('side','?')} | Conf {signal.get('confidence',0):.1f}% | Model: {signal.get('model', 'N/A')}",
        ylabel='Price', addplot=apds, figscale=1.5, returnfig=True,
        hlines=dict(hlines=hlines, colors=colors, linestyle=linestyles, linewidths=1.5, alpha=0.9)
    )
    tmp = NamedTemporaryFile(delete=False, suffix=f"_{symbol}.png")
    fig.savefig(tmp.name, bbox_inches='tight')
    plt.close(fig)
    return tmp.name

# ---------------- Gemini AI Analysis Function ----------------

async def analyze_with_gemini(symbol, data, chart_path=None):
    if not client: return {"side":"none","confidence":0,"reason":"NO_AI_KEY"}

    is_vision_symbol = symbol in VISION_SYMBOLS
    model_to_use = GEMINI_VISION_MODEL if is_vision_symbol and chart_path else GEMINI_TEXT_MODEL

    # --- Data Preparation ---
    c1h = data.get("1h"); candles_4h = data.get("4h")
    df_1h = pd.DataFrame(c1h, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df_1h = df_1h[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    current_price = df_1h['Close'].iloc[-1]
    highs, lows, closes = df_1h['High'].to_numpy(), df_1h['Low'].to_numpy(), df_1h['Close'].to_numpy()
    current_atr = atr(highs, lows, closes)
    last_10_candles_raw = df_1h.tail(10).to_string()

    # --- *** UPDATED PROMPT FOR GEMINI AI *** ---
    # This prompt now specifically asks the AI to focus on chart patterns and candlestick analysis from the image.
    text_prompt = (
        f"You are an expert crypto trading analyst. Your task is to find a high-probability trade setup for {symbol} for the next 24-48 hours. "
        "Your entire analysis MUST be based on the provided chart image combined with the raw text data and option chain data below.\n\n"
        "**Primary Analysis Steps:**\n"
        "1.  **Chart Pattern Analysis (from Image):** First, visually inspect the provided chart. Identify major chart patterns like **Head & Shoulders, Triangles, Channels, Flags, Wedges, and key Support/Resistance levels.**\n"
        "2.  **Candlestick Pattern Analysis (from Image):** Look at the most recent candles on the chart. Identify patterns like **Doji, Engulfing, Hammer, or Morning/Evening Star.**\n"
        "3.  **Data Confirmation (from Text):** Use the raw text data below to confirm your visual analysis with precise numbers (e.g., price levels, volume, RSI, Options OI).\n"
        "4.  **Option Chain Sentiment:** Use the Option Chain data to understand the broader market sentiment. High Call OI walls act as resistance, and high Put OI walls act as support.\n\n"
        "**Raw Market Data for Confirmation:**\n"
        f"- **Current Price:** {current_price:.2f}\n"
        f"- **1H ATR (for SL/TP):** {current_atr:.4f}\n"
        f"- **Options Data:** {data.get('options', {}).get('oi_summary', 'N/A')}\n"
        f"- **Last 10 1H Candles (Raw Data):\n**{last_10_candles_raw}\n\n"
        "**Final Instruction:**\n"
        f"Synthesize all the above points (visual chart patterns + text data + options sentiment) to form a trading decision. "
        f"If confidence is less than {int(SIGNAL_CONF_THRESHOLD)}%, ACTION MUST be 'HOLD' or 'NONE'. "
        f"Otherwise, provide a clear BUY or SELL signal. "
        f"Strictly adhere to the output format: 'ACTION:ACTION_TYPE - ENTRY:x - SL:y - TP:z - TP2:w - REASON:.. - CONF:n%'"
    )

    prompt_parts = []
    if model_to_use == GEMINI_VISION_MODEL:
        try:
            img = Image.open(chart_path)
            prompt_parts = [text_prompt, img]
        except Exception as e:
            print(f"Error reading chart image for {symbol}: {e}. Falling back to text-only.")
            prompt_parts = [text_prompt]
            model_to_use = GEMINI_TEXT_MODEL
    else:
        prompt_parts = [text_prompt]

    try:
        model = genai.GenerativeModel(model_to_use)
        generation_config = genai.types.GenerationConfig(max_output_tokens=800, temperature=0.3)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: model.generate_content(prompt_parts, generation_config=generation_config)
        )
        ai_output = response.text if response.parts else f"ACTION:NONE - REASON:Response blocked by safety filters. - CONF:0%"
        signal = parse_ai_signal(ai_output, symbol, current_atr)
        signal.update({'ai_raw_output': ai_output, 'model': model_to_use})
        return signal
    except Exception as e:
        print(f"Gemini analysis error for {symbol} (Model: {model_to_use}): {e}")
        traceback.print_exc()
        return {"side":"none", "confidence":0, "reason":f"AI_CALL_ERROR: {e}", "model": model_to_use}

# ---------------- Fetch & Telegram ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=20) as r:
            return await r.json() if r.status == 200 else None
    except Exception: return None

async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: print(text); return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try: await session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})
    except Exception as e: print("send_text error:", e)

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: print(caption); return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("chat_id", TELEGRAM_CHAT_ID)
            data.add_field("caption", caption)
            data.add_field("photo", f)
            data.add_field("parse_mode", "Markdown")
            await session.post(url, data=data)
    except Exception as e: print("send_photo error:", e)

# ---------------- Main loop ----------------
async def advanced_options_loop():
    if not client:
        print("🔴 ERROR: Gemini API Key not set. AI analysis will not work.")
        return
    init_redis_plain()
    async with aiohttp.ClientSession() as session:
        startup = f"🤖 FULL Hybrid Bot Started (BTC/ETH: {GEMINI_VISION_MODEL}) • Scan Interval: 30 minutes"
        print(startup); await send_text(session, startup)

        it = 0
        while True:
            it += 1; print(f"\nITER {it} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = time.time()

            all_data = {}
            for sym in SYMBOLS:
                tasks = {
                    "1h": fetch_json(session, CANDLE_URL.format(symbol=sym, tf="1h", limit=CANDLE_LIMITS["1h"])),
                    "options": get_options_data_for_symbol(session, sym)
                }
                results = await asyncio.gather(*tasks.values())
                all_data[sym] = dict(zip(tasks.keys(), results))

            for sym in SYMBOLS:
                chart_path = None
                try:
                    data = all_data.get(sym, {})
                    if not data.get("1h"):
                        print(f"{sym}: Missing critical 1h candle data, skipping"); continue

                    # Generate chart for every analysis attempt
                    chart_path = plot_signal_chart(sym, data["1h"], {"side":"Analyzing...", "confidence":0})

                    final_signal = await analyze_with_gemini(sym, data, chart_path)

                    if final_signal["side"] in ["BUY", "SELL"] and final_signal["confidence"] >= SIGNAL_CONF_THRESHOLD:
                        store_signal(sym, final_signal)
                        model_used = final_signal.get('model', 'N/A')
                        msg = (f"**🔥 AI Trade Alert ({final_signal['confidence']:.1f}%)**\n\n"
                               f"**Asset:** {sym} *(Model: {model_used})*\n"
                               f"**Action:** {final_signal['side']}\n"
                               f"**Entry:** `{fmt_price(final_signal['entry'])}`\n"
                               f"**SL:** `{fmt_price(final_signal['sl'])}` | **TP1:** `{fmt_price(final_signal['tp'])}` | **TP2:** `{fmt_price(final_signal['tp2'])}`\n\n"
                               f"**AI Logic:** _{final_signal['reason']}_")
                        
                        # Re-plot the chart with the final signal details for the alert
                        final_chart_path = plot_signal_chart(sym, data["1h"], final_signal)
                        await send_photo(session, msg, final_chart_path)
                        print(f"⚡ Alert Sent for {sym}: {final_signal['side']} @ {final_signal['entry']:.2f} Conf {final_signal['confidence']:.1f}%")
                        if os.path.exists(final_chart_path): os.remove(final_chart_path)
                    else:
                        print(f"{sym}: No high-confidence trade setup found. (AI says: {final_signal['side']}, Conf: {final_signal['confidence']:.1f}%)")

                except Exception as e:
                    print(f"Error processing {sym}: {e}"); traceback.print_exc()
                finally:
                    if chart_path and os.path.exists(chart_path): os.remove(chart_path)
                    await asyncio.sleep(1) # Rate limit between symbols

            print(f"Processing time: {time.time() - start_time:.2f}s")
            time_to_wait = POLL_INTERVAL - (time.time() - start_time)
            if time_to_wait > 0:
                print(f"Sleeping for {time_to_wait:.0f} seconds...")
                await asyncio.sleep(time_to_wait)

if __name__ == "__main__":
    try:
        asyncio.run(advanced_options_loop())
    except KeyboardInterrupt: print("\nStopped by user.")
