#!/usr/bin/env python3
# main.py - Hybrid GPT-4V (Vision) + GPT-3.5-turbo (Text) Multi-Timeframe Bot

import os, json, asyncio, traceback, time
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from tempfile import NamedTemporaryFile
import re
import base64
from typing import Dict, Any

# plotting (server-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf

# OpenAI client (v0.x client structure, works with both chat and vision)
import openai

# Redis (non-TLS)
import redis

# Load environment variables
load_dotenv()

# ---------------- CONFIG (Hybrid Mode Setup) ----------------
# Symbols: BTC and ETH for analysis
SYMBOLS = ["BTCUSDT", "ETHUSDT"] 

# Symbols that require the more expensive GPT-4 VISION (image analysis). 
# We are currently ONLY using BTCUSDT for Vision analysis to control cost.
VISION_SYMBOLS = ["BTCUSDT"] 
# Ensure GPT-4 model is used for vision/hybrid analysis
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4-turbo") 
TEXT_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo") # Cheaper model for text/ETH analysis

POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", 1800))) # Default 30 mins (1800s)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0))

# API Endpoints (Unchanged)
BASE_URL = "https://api.binance.com"
OPTIONS_BASE_URL = "https://eapi.binance.com"
CANDLE_URL = BASE_URL + "/api/v3/klines?symbol={symbol}&interval={tf}&limit={limit}"
TICKER_24H_URL = BASE_URL + "/api/v3/ticker/24hr?symbol={symbol}"
DEPTH_URL = BASE_URL + "/api/v3/depth?symbol={symbol}&limit=50"
AGGT_URL = BASE_URL + "/api/v3/aggTrades?symbol={symbol}&limit=100"
OPTIONS_TICKER_URL = OPTIONS_BASE_URL + "/eapi/v1/ticker"
CANDLE_LIMITS = {"1h": 999, "4h": 999, "1d": 999}

# --- STABLE PROXY FIX (Unchanged) ---
for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    if var in os.environ:
        del os.environ[var]

# === OpenAI Client Initialization (Unchanged) ===
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    client = True
else:
    client = None
# -------------------------------------------------------------------------

# --- Redis Init/Helpers (Unchanged) ---
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

# ---------------- Options Chain Analysis (Unchanged) ----------------
async def get_options_data_for_symbol(session, base_symbol) -> Dict[str, Any]:
    # (Function body is the same as the previous correct implementation)
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

    total_call_oi = 0
    total_put_oi = 0
    all_ivs = []
    
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
    
    if total_oi == 0:
        sentiment = "No Open Interest."
    elif pcr < 0.7:
        sentiment = "Strong Call Bias (Bullish sentiment) - PCR < 0.7."
    elif pcr > 1.2:
        sentiment = "Strong Put Bias (Bearish/Hedging sentiment) - PCR > 1.2."
    else:
        sentiment = f"Neutral to Moderate Sentiment (PCR: {pcr:.2f})."
        
    data["options_sentiment"] = sentiment
    data["oi_summary"] = (
        f"Next Expiry Date: {datetime.fromtimestamp(next_expiry_date_ms/1000).strftime('%Y-%m-%d %H:%M')}. "
        f"Total OI: {total_oi:.2f}. Call OI: {total_call_oi:.2f}. Put OI: {total_put_oi:.2f}. "
        f"Put/Call Ratio (PCR): {pcr:.2f}. "
        f"Average Implied Volatility (IV): {avg_iv * 100:.2f}%."
    )
    data["near_term_symbol"] = next_expiry_contracts[0]['symbol'] if next_expiry_contracts else 'N/A'
    return data

# ---------------- Hybrid GPT Analysis Function ----------------

async def analyze_with_openai(symbol, data, chart_path=None): 
    if not client: return {"side":"none","confidence":0,"reason":"NO_AI_KEY"}
    
    # Select Model based on symbol
    is_vision_symbol = symbol in VISION_SYMBOLS
    model_to_use = VISION_MODEL if is_vision_symbol else TEXT_MODEL
    
    # --- Data Preparation (Same as before) ---
    c1h = data.get("1h")
    candles_4h = data.get("4h")
    spot_depth = data.get("depth")
    agg_trades = data.get("aggTrades")
    
    df_1h = pd.DataFrame(c1h, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df_1h = df_1h[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    current_price = df_1h['Close'].iloc[-1]
    
    last_3_candles = df_1h.tail(3)
    c1, c2, c3 = last_3_candles.iloc[-3], last_3_candles.iloc[-2], last_3_candles.iloc[-1]
    pattern_summary = "No clear pattern."
    if c3['Close'] > c3['Open'] and c2['Close'] < c2['Open'] and c1['Close'] < c1['Open'] and c3['Open'] > c2['Close']:
        pattern_summary = "Possible Bullish Engulfing/Morning Star formation."
    elif c3['Close'] < c3['Open'] and c2['Close'] > c2['Open'] and c1['Close'] > c1['Open'] and c3['Open'] < c2['Close']:
        pattern_summary = "Possible Bearish Engulfing/Evening Star formation."
        
    highs = df_1h['High'].to_numpy(); lows = df_1h['Low'].to_numpy(); closes = df_1h['Close'].to_numpy()
    current_atr = atr(highs, lows, closes) 
    
    # 4H Data
    if candles_4h:
        df_4h = pd.DataFrame(candles_4h, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
        df_4h = df_4h[['Open', 'High', 'Low', 'Close']].astype(float)
        long_term_ema = df_4h['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        long_term_rsi = df_4h['Close'].rolling(window=14).apply(lambda x: rsi(x), raw=True).iloc[-1]
        if current_price > long_term_ema:
            long_term_trend = f"BULLISH (Price above 4H EMA 200: {long_term_ema:.2f})"
        else:
            long_term_trend = f"BEARISH (Price below 4H EMA 200: {long_term_ema:.2f})"
        recent_4h_high = df_4h['High'].iloc[-50:].max()
        recent_4h_low = df_4h['Low'].iloc[-50:].min()
    else:
        long_term_trend = "4H Data Unavailable"
        long_term_rsi = 50.0 
        recent_4h_high = current_price
        recent_4h_low = current_price

    # Spot Flow
    bid_vol_spot = sum([float(b[1]) for b in spot_depth.get('bids', [])])
    ask_vol_spot = sum([float(a[1]) for a in spot_depth.get('asks', [])])
    spot_imbalance = (bid_vol_spot - ask_vol_spot) / (bid_vol_spot + ask_vol_spot) * 100 if (bid_vol_spot + ask_vol_spot) > 0 else 0
    buy_vol_agg = sum([float(t['q']) for t in agg_trades if not t['m']])
    sell_vol_agg = sum([float(t['q']) for t in agg_trades if t['m']])
    agg_imbalance = (buy_vol_agg - sell_vol_agg) / (buy_vol_agg + sell_vol_agg) * 100 if (buy_vol_agg + sell_vol_agg) > 0 else 0
    
    # Options Data
    opt_info = data.get('options')
    opt_summary = opt_info.get('oi_summary', 'No Open Interest Data.')
    opt_sentiment = opt_info.get('options_sentiment', 'Neutral')
    pcr_value = re.search(r'PCR:\s*([\d.]+)', opt_summary)
    pcr_val = float(pcr_value.group(1)) if pcr_value else 0.0

    # 4. Construct the Detailed Prompt
    sys_prompt = (
        "You are a sophisticated quantitative crypto analyst using a comprehensive multi-factor, multi-timeframe model. "
        "Your task is to analyze the provided market data and generate a high-conviction trading signal "
        "(BUY/SELL/HOLD) for the next 24-48 hours. Use ALL data points below, especially combining Macro (4H/OI) "
        "with Short-Term (1H/Flow) to formulate your logic. "
        f"Strictly adhere to the output format: 'SYMBOL - ACTION - ENTRY:x - SL:y - TP:z - TP2:w - REASON:.. - CONF:n%'"
    )
    
    # --- Message Content (TEXT PART) ---
    text_content = f"""
    ### {symbol} - Advanced Multi-Timeframe Analysis Request
    
    **1. Macro Trend (4H TF - Last 999 Candles):**
    - Long-Term Trend (EMA 200): {long_term_trend}
    - 4H RSI (14-period): {long_term_rsi:.2f} (Overbought > 70, Oversold < 30)
    - Recent 4H Key Resistance: {recent_4h_high:.2f} | Recent 4H Key Support: {recent_4h_low:.2f}
    
    **2. Price Action & Momentum (1H TF - Short Term):**
    - Current Price: {current_price:.2f}
    - Last 3 Candles (OHLCV): \n{last_3_candles.to_string()}
    - Recent Candlestick Pattern: {pattern_summary}
    - Current ATR (14-period): {current_atr:.4f} (Use this for SL/TP1 suggestions: SL ~1.5x ATR, TP1 ~3x ATR)
    
    **3. Spot Market Flow Analysis:**
    - Spot Depth Imbalance: {spot_imbalance:.2f}% 
    - Aggressive Trades Imbalance: {agg_imbalance:.2f}% 
    
    **4. Options Chain Deep Analysis (Macro Sentiment & Key Levels):**
    - Synthesized Sentiment: {opt_sentiment}
    - Put/Call Ratio (PCR): {pcr_val:.2f}.
    - OI Summary (Reference for Macro S/R/Expiry): {opt_summary}
    - **Crucial Interpretation Hint:** The strongest Call OI walls act as Major Resistance (potential TP2), and the strongest Put OI walls act as Major Support (potential SL).
    
    **5. Trading Instructions (Set TP1 and TP2):**
    - **TP1 (Short-Term):** Must be based on the current ATR and short-term 1H momentum (3x ATR default).
    - **TP2 (Long-Term/Macro):** Must be set based on **4H Key S/R levels** or the **strongest unbreached Options OI Wall**.
    - **Fake Breakout Check:** If the current price is near or breaking {recent_4h_high:.2f} or {recent_4h_low:.2f}, confirm the move with strong **Aggressive Flow** (> 20%) and **OI confirmation**. If confirmation is missing, signal HOLD/NONE or suggest a reversal.
    - **CRITICAL RULE (High Confirmation Requirement):** If your combined analysis leads to a confidence of less than {int(SIGNAL_CONF_THRESHOLD)}%, the **ACTION MUST be 'HOLD' or 'NONE'**, regardless of the market direction.
    - **If Confidence is ≥ {int(SIGNAL_CONF_THRESHOLD)}%**, provide a specific **BUY/SELL** signal with calculated ENTRY, SL, TP, and TP2.
    """

    # --- Construct Message for GPT ---
    messages = [{"role": "system", "content": sys_prompt}]
    
    if is_vision_symbol and chart_path:
        try:
            with open(chart_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            
            # Message structure for GPT-4V (Image + Text)
            messages.append({"role": "user", "content": [
                {"type": "text", "text": "Analyze the following 1H Candlestick Chart (with 200 SMA) and combine its pattern recognition with the detailed market data below to generate the final signal. **Prioritize pattern confirmation from the chart.**\n\n--- DETAILED MARKET DATA ---\n" + text_content},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]})
            
        except Exception as e:
            print(f"Error reading chart image for {symbol}: {e}. Falling back to Text-Only mode.")
            messages.append({"role": "user", "content": text_content})
            is_vision_symbol = False # Fallback 
            model_to_use = TEXT_MODEL # Fallback
            
    else:
        # Message structure for Text-Only (GPT-3.5-turbo or GPT-4-turbo Text)
        messages.append({"role": "user", "content": text_content})

    # 5. Call OpenAI API
    try:
        loop = asyncio.get_running_loop()
        def call(): 
            return openai.ChatCompletion.create(
                model=model_to_use,
                messages=messages,
                max_tokens=800 if is_vision_symbol else 600, # More tokens for Vision output
                temperature=0.3
            )
        resp = await loop.run_in_executor(None, call)
        ai_output = resp.choices[0].message.content.strip()
        
        # 6. Parse the AI Output
        signal = parse_ai_signal(ai_output, symbol, current_atr)
        signal['ai_raw_output'] = ai_output
        signal['model'] = model_to_use # Add model info
        return signal
        
    except Exception as e:
        print(f"OpenAI analysis error for {symbol} (Model: {model_to_use}): {e}")
        return {"side":"none","confidence":0,"reason":f"AI_CALL_ERROR ({model_to_use}): {str(e)}", "model": model_to_use}

# ---------------- Utility Functions (Unchanged) ----------------
# rsi, atr, fmt_price, parse_ai_signal (These remain the same as they handle math and output formatting)

# --- plot_signal_chart (Minor change for dynamic filename/cleanup) ---
def plot_signal_chart(symbol, candles, signal):
    df_candles = pd.DataFrame(candles, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df_candles['OpenTime'] = pd.to_datetime(df_candles['OpenTime'], unit='ms')
    df_candles = df_candles.set_index(pd.DatetimeIndex(df_candles['OpenTime']))
    df_candles = df_candles[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    df_candles['SMA200'] = df_candles['Close'].rolling(window=200).mean()
    df_plot = df_candles.iloc[-300:] 

    apds = [
        mpf.make_addplot(df_plot['SMA200'], color='#FFA500', panel=0, width=1.0, secondary_y=False), 
    ]
    hlines = []; colors = []; linestyles = []
    
    if signal["side"] in ["BUY", "SELL"]:
        hlines.extend([signal["entry"], signal["sl"], signal["tp"]])
        colors = ['blue', 'red', 'green']
        linestyles = ['-', '--', '--'] 
        
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo', gridcolor='#e0e0e0', facecolor='#ffffff', figcolor='#ffffff', y_on_right=False,
        marketcolors=mpf.make_marketcolors(up='g', down='r', edge='inherit', wick='inherit', volume='inherit')
    )

    fig, axlist = mpf.plot(
        df_plot, type='candle', style=s,
        title=f"{symbol} 1H Advanced Signal ({signal.get('side','?')}) | Conf {signal.get('confidence',0)}% | Model: {signal.get('model', 'N/A')}",
        ylabel='Price', addplot=apds, figscale=1.5, returnfig=True,         
        hlines=dict(hlines=hlines, colors=colors, linestyle=linestyles, linewidths=1.5, alpha=0.9), 
    )

    ax = axlist[0] 
    logo_text = "⚡ Flying Raijin - Multi-TF Model ⚡"
    ax.text(
        0.99, 0.01, logo_text, transform=ax.transAxes, fontsize=14, fontweight='bold',
        color='#FFD700', ha='right', va='bottom', alpha=0.8,
        bbox=dict(facecolor='#333333', alpha=0.7, edgecolor='#FFD700', linewidth=1, boxstyle='round,pad=0.5')
    )

    # Use NamedTemporaryFile to ensure unique path and safe file handling
    tmp = NamedTemporaryFile(delete=False, suffix=f"_{symbol}.png")
    fig.savefig(tmp.name, bbox_inches='tight')
    plt.close(fig)
    return tmp.name

# ---------------- Fetch & Telegram (Unchanged) ----------------
async def fetch_json(session,url):
    try:
        async with session.get(url,timeout=20) as r:
            if r.status!=200: return None
            return await r.json()
    except Exception: return None

async def send_text(session,text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: print(text); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try: await session.post(url,json={"chat_id":TELEGRAM_CHAT_ID,"text":text, "parse_mode": "Markdown"})
    except Exception as e: print("send_text error:", e)

async def send_photo(session,caption,path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: print(caption); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path,"rb") as f:
            data=aiohttp.FormData(); data.add_field("chat_id",TELEGRAM_CHAT_ID); data.add_field("caption",caption); data.add_field("photo",f); data.add_field("parse_mode", "Markdown")
            await session.post(url,data=data)
    except Exception as e: print("send_photo error:", e)


# ---------------- Main loop (Updated for Hybrid Logic) ----------------
async def advanced_options_loop():
    if not client:
        print("🔴 ERROR: OpenAI API Key (OPENAI_API_KEY) not set. AI analysis will not work.")
        return
        
    init_redis_plain() 
    
    async with aiohttp.ClientSession() as session:
        startup=f"🤖 Hybrid Bot Started (BTC: {VISION_MODEL} Vision | ETH: {TEXT_MODEL}) • Poll {POLL_INTERVAL//60}min"
        print(startup); await send_text(session,startup)
        
        while True:
            it=1; print(f"\nITER {it} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = time.time()
            
            # --- Fetch all market data concurrently ---
            tasks = {}
            for sym in SYMBOLS:
                tasks[f"{sym}_24h"] = fetch_json(session, TICKER_24H_URL.format(symbol=sym))
                tasks[f"{sym}_depth"] = fetch_json(session, DEPTH_URL.format(symbol=sym))
                tasks[f"{sym}_aggTrades"] = fetch_json(session, AGGT_URL.format(symbol=sym))
                tasks[f"{sym}_1h"] = fetch_json(session, CANDLE_URL.format(symbol=sym, tf="1h", limit=CANDLE_LIMITS["1h"]))
                tasks[f"{sym}_4h"] = fetch_json(session, CANDLE_URL.format(symbol=sym, tf="4h", limit=CANDLE_LIMITS["4h"]))
                tasks[f"{sym}_options"] = get_options_data_for_symbol(session, sym) 

            results = await asyncio.gather(*tasks.values())
            
            fetched_data = {}; 
            for key, res in zip(tasks.keys(), results):
                sym, key_type = key.split('_', 1)
                if sym not in fetched_data: fetched_data[sym] = {}
                fetched_data[sym][key_type] = res

            # --- Process each symbol ---
            for sym in SYMBOLS:
                chart_path = None
                try:
                    data = fetched_data.get(sym, {})
                    c1h = data.get("1h"); 
                    spot_depth = data.get("depth"); agg_data = data.get("aggTrades"); 
                    
                    if not all([c1h, spot_depth, agg_data]):
                        print(f"{sym}: Missing critical data, skipping"); continue

                    # 1. Generate Chart Image if Vision is enabled for the symbol
                    if sym in VISION_SYMBOLS:
                        # Temporary signal object just to generate chart with title (side/conf will be updated later)
                        chart_path = plot_signal_chart(sym, c1h, {"side":"N/A", "confidence":0, "model": VISION_MODEL})

                    # 2. Run Hybrid AI Analysis
                    final_signal = await analyze_with_openai(sym, data, chart_path)
                    
                    # 3. Re-plot chart with final signal if it was a Vision analysis (to update title/lines)
                    # For a hybrid setup, re-plotting is crucial if the initial plot didn't contain the final signal.
                    # We will re-plot if a signal was generated, regardless of model, to get correct SL/TP lines on chart.
                    
                    # 4. Process the Signal
                    if final_signal["side"] in ["BUY", "SELL"] and final_signal["confidence"] >= SIGNAL_CONF_THRESHOLD:
                        store_signal(sym, final_signal)
                        
                        # Use the correct model name in the final Telegram message
                        model_used = final_signal.get('model', TEXT_MODEL)
                        
                        msg=f"**🔥 Advanced AI Signal ({final_signal['confidence']}%)**\n\n**Asset:** {sym} *(Model: {model_used})*\n**Action:** {final_signal['side']}\n**Entry:** `{fmt_price(final_signal['entry'])}`\n**SL:** `{fmt_price(final_signal['sl'])}`\n**TP1:** `{fmt_price(final_signal['tp'])}` | **TP2 (Macro):** `{fmt_price(final_signal['tp2'])}`\n\n**AI Logic:** {final_signal['reason']}"
                        
                        # Re-plot the chart with the final calculated lines and model info in title
                        chart_path = plot_signal_chart(sym, c1h, final_signal)
                        
                        await send_photo(session,msg,chart_path)
                        print("⚡",msg.replace('\n', ' '))
                    else:
                        print(f"{sym}: AI suggested HOLD/NONE or Confidence too low ({final_signal['confidence']}%). Model: {final_signal.get('model', 'N/A')}. Reason: {final_signal['reason'][:50]}...")
                        
                except Exception as e:
                    print(f"Error processing {sym}: {e}")
                    traceback.print_exc()
                finally:
                    # Clean up the chart image file
                    if chart_path and os.path.exists(chart_path):
                        os.remove(chart_path)
                
            print(f"Processing time: {time.time() - start_time:.2f}s")
            await asyncio.sleep(POLL_INTERVAL)

if __name__=="__main__":
    try: 
        print("Connecting to Redis...")
        asyncio.run(advanced_options_loop())
    except KeyboardInterrupt: print("Stopped by user")
