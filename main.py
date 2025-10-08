#!/usr/bin/env python3
# main.py - Advanced Multi-Timeframe BTC/ETH Bot (999 Candles, 4H/1H Analysis, GPT-3.5-turbo)

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

# OpenAI client
import openai 

# Redis (non-TLS)
import redis

# Load environment variables
load_dotenv()

# ---------------- CONFIG ----------------
SYMBOLS = ["BTCUSDT", "ETHUSDT"] 
POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", 1800))) 
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo") # Changed to v0.x supported model
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0)) 

# API Endpoints
BASE_URL = "https://api.binance.com" 
OPTIONS_BASE_URL = "https://eapi.binance.com" 

CANDLE_URL = BASE_URL + "/api/v3/klines?symbol={symbol}&interval={tf}&limit={limit}"
TICKER_24H_URL = BASE_URL + "/api/v3/ticker/24hr?symbol={symbol}"
DEPTH_URL = BASE_URL + "/api/v3/depth?symbol={symbol}&limit=50" 
AGGT_URL = BASE_URL + "/api/v3/aggTrades?symbol={symbol}&limit=100" 

# Options Endpoints
OPTIONS_TICKER_URL = OPTIONS_BASE_URL + "/eapi/v1/ticker" 

# Set all Timeframes to the maximum robust limit of 999 klines (Binance API limit is 1000)
CANDLE_LIMITS = {"1h": 999, "4h": 999, "1d": 999} 

# --- STABLE PROXY FIX: Remove environment variables that might pass 'proxies' argument automatically ---
for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    if var in os.environ:
        del os.environ[var]
# ----------------------------------------------------------------------------------------------------

# === FIX for client initialization (Uses openai v0.x Client structure) ===
if OPENAI_API_KEY:
    # Set the key globally for v0.x functions
    openai.api_key = OPENAI_API_KEY
    client = True # Flag to check if the key is available
else:
    client = None
# -------------------------------------------------------------------------

# --- Redis Init/Helpers (No changes) ---
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
    
    target_tickers = [t for t in all_tickers if t.get('underlying') == base_symbol]
    if not target_tickers: return data

    all_expiries = sorted(list(set(t.get('expiryDate', 0) for t in target_tickers)))
    next_expiry_date_ms = next(
        (exp for exp in all_expiries if exp > current_time_ms), 
        None
    )
    if not next_expiry_date_ms:
        data['oi_summary'] = f"Warning: No future option expiries found for {base_symbol}."
        return data

    next_expiry_contracts = [
        t for t in target_tickers 
        if t.get('expiryDate') == next_expiry_date_ms
    ]
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


# ---------------- GPT-3.5-turbo Analysis (v0.x Model) ----------------

async def analyze_with_openai(symbol, data): 
    # Use the client flag check
    if not client: return {"side":"none","confidence":0,"reason":"NO_AI_KEY"}
    
    # Extract data parts
    c1h = data.get("1h")
    spot_depth = data.get("depth")
    agg_trades = data.get("aggTrades")
    options_data = data.get("options")
    candles_4h = data.get("4h")

    # 1. Prepare 1H Data (Short-Term)
    df_1h = pd.DataFrame(c1h, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df_1h = df_1h[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    current_price = df_1h['Close'].iloc[-1]
    
    # Simple Candlestick Pattern detection
    last_3_candles = df_1h.tail(3)
    c1, c2, c3 = last_3_candles.iloc[-3], last_3_candles.iloc[-2], last_3_candles.iloc[-1]
    pattern_summary = "No clear pattern."
    if c3['Close'] > c3['Open'] and c2['Close'] < c2['Open'] and c1['Close'] < c1['Open'] and c3['Open'] > c2['Close']:
        pattern_summary = "Possible Bullish Engulfing/Morning Star formation."
    elif c3['Close'] < c3['Open'] and c2['Close'] > c2['Open'] and c1['Close'] > c1['Open'] and c3['Open'] < c2['Close']:
        pattern_summary = "Possible Bearish Engulfing/Evening Star formation."
        
    # ATR for Risk Management
    highs = df_1h['High'].to_numpy(); lows = df_1h['Low'].to_numpy(); closes = df_1h['Close'].to_numpy()
    current_atr = atr(highs, lows, closes) 
    
    # ----------------- 4H Long-Term Analysis (Macro) -----------------
    if candles_4h:
        df_4h = pd.DataFrame(candles_4h, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
        df_4h = df_4h[['Open', 'High', 'Low', 'Close']].astype(float)
        
        long_term_ema = df_4h['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        
        # === FIX: Removed .values because raw=True passes numpy.ndarray ===
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

    # 2. Spot Market Flow Analysis
    bid_vol_spot = sum([float(b[1]) for b in spot_depth.get('bids', [])])
    ask_vol_spot = sum([float(a[1]) for a in spot_depth.get('asks', [])])
    spot_imbalance = (bid_vol_spot - ask_vol_spot) / (bid_vol_spot + ask_vol_spot) * 100 if (bid_vol_spot + ask_vol_spot) > 0 else 0
    
    # Aggressive Trades Analysis
    buy_vol_agg = sum([float(t['q']) for t in agg_trades if not t['m']])
    sell_vol_agg = sum([float(t['q']) for t in agg_trades if t['m']])
    agg_imbalance = (buy_vol_agg - sell_vol_agg) / (buy_vol_agg + sell_vol_agg) * 100 if (buy_vol_agg + sell_vol_agg) > 0 else 0
    
    # 3. Options Data Summary 
    opt_info = data.get('options')
    opt_summary = opt_info.get('oi_summary', 'No Open Interest Data.')
    opt_sentiment = opt_info.get('options_sentiment', 'Neutral')
    opt_symbol = opt_info.get('near_term_symbol', 'N/A')
    pcr_value = re.search(r'PCR:\s*([\d.]+)', opt_summary)
    pcr_val = float(pcr_value.group(1)) if pcr_value else 0.0

    # 4. Construct the Detailed Prompt
    sys_prompt = (
        "You are a sophisticated quantitative crypto analyst using a comprehensive multi-factor, multi-timeframe model. "
        "Your task is to analyze the provided market data and generate a high-conviction trading signal "
        "(BUY/SELL/HOLD) for the next 24-48 hours. Use ALL data points below, especially combining Macro (4H/OI) "
        "with Short-Term (1H/Flow) to formulate your logic. "
        "Strictly adhere to the output format, providing two Take Profit levels: 'SYMBOL - ACTION - ENTRY:x - SL:y - TP:z - TP2:w - REASON:.. - CONF:n%'"
    )

    usr_prompt = f"""
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
    - **If Confidence is ≥ {int(SIGNAL_CONF_THRESHOLD)}%**, provide a specific **BUY/SELL** signal with calculated ENTRY, SL, TP, and TP2.
    - **If Confidence is < {int(SIGNAL_CONF_THRESHOLD)}%**, the action must be **HOLD** or **NONE**.
    """

    try:
        loop = asyncio.get_running_loop()
        def call(): 
            # In v0.x, use completion.create for chat endpoint compatibility
            return openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": usr_prompt}
                ],
                max_tokens=600,
                temperature=0.3
            )
        # Note: v0.x client methods are not bound to the client instance for ChatCompletion
        resp = await loop.run_in_executor(None, call)
        ai_output = resp.choices[0].message.content.strip()
        
        # 3. Parse the AI Output
        signal = parse_ai_signal(ai_output, symbol, current_atr)
        signal['ai_raw_output'] = ai_output
        return signal
        
    except Exception as e:
        print(f"OpenAI analysis error for {symbol}: {e}")
        return {"side":"none","confidence":0,"reason":f"AI_CALL_ERROR: {str(e)}"}


# ---------------- Utility Functions ----------------

def rsi(prices, period=14):
    """Calculates Relative Strength Index (RSI) using Pandas for standalone calculation."""
    # Ensure prices is a pandas Series for the original logic to work correctly inside the standalone RSI function
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
        
    if len(prices) < period: return 50.0
    df = pd.DataFrame(prices, columns=['Close'])
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    # Check for division by zero before calculating rs
    # If loss is zero, rs should be infinity (or a large number) for rsi to be 100
    rs = np.divide(gain, loss, out=np.full_like(gain, np.inf), where=loss != 0)
    
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.iloc[-1] if not rsi_val.empty else 50.0

def atr(highs, lows, closes, period=14):
    if len(closes) < period: return 0.0
    tr = []
    for i in range(1, len(closes)):
        tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    if len(tr) < period: return 0.0
    initial_atr = np.mean(tr[:period])
    atrs = [initial_atr]
    for i in range(period, len(tr)):
        atrs.append((atrs[-1] * (period - 1) + tr[i]) / period)
    return atrs[-1] if atrs else 0.0

def fmt_price(p): return f"{p:.4f}" if abs(p)<1 else f"{p:.2f}"

def parse_ai_signal(ai_output, symbol, current_atr):
    signal = {"side": "none", "confidence": 0, "reason": ai_output, "entry": 0, "sl": 0, "tp": 0, "tp2": 0} 
    
    conf_match = re.search(r'CONF:(\d+)%', ai_output, re.IGNORECASE)
    if conf_match:
        signal['confidence'] = int(conf_match.group(1))

    action_match = re.search(r'(\w+)\s*-\s*(BUY|SELL|HOLD|NONE)', ai_output, re.IGNORECASE)
    entry_match = re.search(r'ENTRY:([\d.]+)', ai_output, re.IGNORECASE)
    sl_match = re.search(r'SL:([\d.]+)', ai_output, re.IGNORECASE)
    tp_match = re.search(r'TP:([\d.]+)', ai_output, re.IGNORECASE)
    tp2_match = re.search(r'TP2:([\d.]+)', ai_output, re.IGNORECASE) 
    reason_match = re.search(r'REASON:(.*?)(\s*-\s*CONF:|\s*$)', ai_output, re.IGNORECASE | re.DOTALL)
    
    if action_match:
        side = action_match.group(2).upper()
        signal['side'] = side
        
        if side in ["BUY", "SELL"]:
            try:
                entry = float(entry_match.group(1)) if entry_match else 0
                sl = float(sl_match.group(1)) if sl_match else 0
                tp = float(tp_match.group(1)) if tp_match else 0
                tp2 = float(tp2_match.group(1)) if tp2_match else 0 
                
                if entry == 0: 
                    price_match = re.search(r'Current Price: ([\d.]+)', signal['reason'])
                    if price_match: entry = float(price_match.group(1))
                    else: entry = 0

                if entry == 0: raise ValueError("Entry price could not be determined.")

                if side == "BUY":
                    signal['entry'] = entry
                    signal['sl'] = sl if sl > 0 and sl < entry else entry - (current_atr * 1.5)
                    signal['tp'] = tp if tp > entry else entry + (current_atr * 3.0) 
                    signal['tp2'] = tp2 if tp2 > entry else entry + (current_atr * 4.5) # Default TP2
                elif side == "SELL":
                    signal['entry'] = entry
                    signal['sl'] = sl if sl > entry else entry + (current_atr * 1.5) 
                    signal['tp'] = tp if tp < entry and tp > 0 else entry - (current_atr * 3.0) 
                    signal['tp2'] = tp2 if tp2 < entry and tp2 > 0 else entry - (current_atr * 4.5) # Default TP2
                
            except Exception as e:
                signal['reason'] += f" | (Parsing Error: {e})"
                signal['side'] = "none" 

    if reason_match:
        signal['reason'] = reason_match.group(1).strip()
        
    return signal

# ---------------- Professional Candlestick Chart ----------------

def plot_signal_chart(symbol, candles, signal):
    
    df_candles = pd.DataFrame(candles, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df_candles['OpenTime'] = pd.to_datetime(df_candles['OpenTime'], unit='ms')
    df_candles = df_candles.set_index(pd.DatetimeIndex(df_candles['OpenTime']))
    df_candles = df_candles[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    df_candles['SMA200'] = df_candles['Close'].rolling(window=200).mean()
    
    # Filter to the last 300 points for stable chart plotting
    df_plot = df_candles.iloc[-300:] 

    apds = [
        mpf.make_addplot(df_plot['SMA200'], color='#FFA500', panel=0, width=1.0, secondary_y=False), 
    ]

    hlines = []
    colors = []; linestyles = []
    
    if signal["side"] in ["BUY", "SELL"]:
        # Only plot Entry, SL, and TP1 on the chart
        hlines.extend([signal["entry"], signal["sl"], signal["tp"]])
        colors = ['blue', 'red', 'green']
        linestyles = ['-', '--', '--'] 
        
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo', 
        gridcolor='#e0e0e0',    
        facecolor='#ffffff',    
        figcolor='#ffffff',     
        y_on_right=False,       
        marketcolors=mpf.make_marketcolors(
            up='g', down='r', edge='inherit', wick='inherit', volume='inherit'    
        )
    )

    fig, axlist = mpf.plot(
        df_plot, 
        type='candle',
        style=s,
        title=f"{symbol} 1H Advanced Signal ({signal.get('side','?')}) | Conf {signal.get('confidence',0)}%",
        ylabel='Price',
        addplot=apds,           
        figscale=1.5,           
        returnfig=True,         
        hlines=dict(hlines=hlines, colors=colors, linestyle=linestyles, linewidths=1.5, alpha=0.9), 
    )

    ax = axlist[0] 
    logo_text = "⚡ Flying Raijin - Multi-TF Model ⚡"
    
    ax.text(
        0.99, 0.01, logo_text, transform=ax.transAxes, fontsize=14, fontweight='bold',
        color='#FFD700', ha='right', va='bottom', alpha=0.8,
        bbox=dict(facecolor='#333333', alpha=0.7, edgecolor='#FFD700', linewidth=1, boxstyle='round,pad=0.5')
    )

    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches='tight')
    plt.close(fig)
    return tmp.name

# ---------------- Fetch & Telegram ----------------
async def fetch_json(session,url):
    try:
        async with session.get(url,timeout=20) as r:
            if r.status!=200: 
                print(f"API Error ({r.status}): {url}")
                return None
            return await r.json()
    except Exception as e:
        print("fetch_json error:", e)
        return None

async def send_text(session,text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try: await session.post(url,json={"chat_id":TELEGRAM_CHAT_ID,"text":text, "parse_mode": "Markdown"})
    except Exception as e: print("send_text error:", e)

async def send_photo(session,caption,path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(caption); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path,"rb") as f:
            data=aiohttp.FormData(); data.add_field("chat_id",TELEGRAM_CHAT_ID); data.add_field("caption",caption); data.add_field("photo",f); data.add_field("parse_mode", "Markdown")
            await session.post(url,data=data)
    except Exception as e: print("send_photo error:", e)


# ---------------- Main loop ----------------
async def advanced_options_loop():
    if not client:
        print("🔴 ERROR: OpenAI API Key (OPENAI_API_KEY) not set. AI analysis will not work.")
        return
        
    init_redis_plain() 
    
    async with aiohttp.ClientSession() as session:
        startup=f"🤖 Advanced BTC/ETH Bot Started (Multi-TF/OI Analysis, GPT-3.5-turbo) • Poll {POLL_INTERVAL//60}min"
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
                
                # Fetch 1H (Short Term) and 4H (Long Term) Candles (999 limit)
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
                try:
                    data = fetched_data.get(sym, {})
                    c1h = data.get("1h"); 
                    spot_depth = data.get("depth"); agg_data = data.get("aggTrades"); 
                    
                    if not all([c1h, spot_depth, agg_data]):
                        print(f"{sym}: Missing critical data, skipping"); continue
                    
                    # 1. Run Advanced AI Analysis (passing entire data dict)
                    final_signal = await analyze_with_openai(sym, data)
                    
                    if final_signal["side"] in ["BUY", "SELL"]:
                        
                        if final_signal["confidence"] >= SIGNAL_CONF_THRESHOLD:
                            store_signal(sym, final_signal)

                            # Message updated to include TP1 and TP2
                            msg=f"**🔥 Advanced AI Signal ({final_signal['confidence']}%)**\n\n**Asset:** {sym}\n**Action:** {final_signal['side']}\n**Entry:** `{fmt_price(final_signal['entry'])}`\n**SL:** `{fmt_price(final_signal['sl'])}`\n**TP1:** `{fmt_price(final_signal['tp'])}` | **TP2 (Macro):** `{fmt_price(final_signal['tp2'])}`\n\n**AI Logic:** {final_signal['reason']}"
                            
                            chart=plot_signal_chart(sym, c1h, final_signal)
                            await send_photo(session,msg,chart)
                            print("⚡",msg.replace('\n', ' '))
                        else:
                             print(f"{sym}: AI signal generated but confidence too low ({final_signal['confidence']}%) - Action: {final_signal['side']}")
                    else:
                        print(f"{sym}: AI suggested HOLD/NONE. Conf: {final_signal['confidence']}%. Reason: {final_signal['reason'][:50]}...")
                        
                except Exception as e:
                    print(f"Error processing {sym}: {e}")
                    traceback.print_exc()
                    
            print(f"Processing time: {time.time() - start_time:.2f}s")
            await asyncio.sleep(POLL_INTERVAL)

if __name__=="__main__":
    try: 
        print("Connecting to Redis...")
        asyncio.run(advanced_options_loop())
    except KeyboardInterrupt: print("Stopped by user")
