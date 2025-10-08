#!/usr/bin/env python3
# main.py - Professional BTC/ETH Bot (Options Chain, Deep Candlesticks, GPT-4o-mini Analysis)
# 'Flying Raijin' Logo Included

import os, json, asyncio, traceback, time
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from tempfile import NamedTemporaryFile
import re # For simple signal parsing

# plotting (server-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf 

# OpenAI client
from openai import OpenAI

# Redis (non-TLS)
import redis

# Load environment variables (API keys, tokens, etc.)
load_dotenv()

# ---------------- CONFIG ----------------
# Bot will ONLY focus on BTC and ETH for Options Chain Analysis
SYMBOLS = ["BTCUSDT", "ETHUSDT"] 

POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", 1800))) # Reduced polling for comprehensive analysis (30 minutes)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini") # Using GPT-4o-mini as requested
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0)) # Higher confidence needed

# API Endpoints
BASE_URL = "https://api.binance.com" 
OPTIONS_BASE_URL = "https://eapi.binance.com" # Vanilla Options

CANDLE_URL = BASE_URL + "/api/v3/klines?symbol={symbol}&interval={tf}&limit={limit}"
TICKER_24H_URL = BASE_URL + "/api/v3/ticker/24hr?symbol={symbol}"
DEPTH_URL = BASE_URL + "/api/v3/depth?symbol={symbol}&limit=50" # Spot Depth
AGGT_URL = BASE_URL + "/api/v3/aggTrades?symbol={symbol}&limit=100" # Spot AggTrades

# Options Endpoints
OPTIONS_EXCHANGE_INFO_URL = OPTIONS_BASE_URL + "/eapi/v1/exchangeInfo"
OPTIONS_DEPTH_URL = OPTIONS_BASE_URL + "/eapi/v1/depth?symbol={symbol}&limit=500" # Options Chain Depth

# Candles config: 720 candles for 1h TF
CANDLE_LIMITS = {"1h": 720, "4h": 200, "1d": 200}

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --- Redis Init/Helpers (For state management) ---
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

# ---------------- Options Chain Analysis (FIXED FUNCTION) ----------------
# FIX: Changed filtering logic from 'baseAsset' to 'underlying' to prevent KeyError.
async def get_options_data_for_symbol(session, base_symbol):
    data = {"exchange_info": None, "put_call_ratio": "N/A", "implied_volatility": "N/A", "near_term_symbol": None}
    
    # 1. Get all available options symbols for the base asset (e.g., BTCUSDT)
    exchange_info = await fetch_json(session, OPTIONS_EXCHANGE_INFO_URL)
    if not exchange_info or 'optionSymbols' not in exchange_info: 
        return data
    
    data["exchange_info"] = exchange_info
    
    # Use the 'underlying' key for filtering, which is the correct symbol (e.g., BTCUSDT) 
    # for contract symbols in Binance Options API.
    matching_contracts = [
        sym for sym in exchange_info.get('optionSymbols', [])
        if sym.get('underlying') == base_symbol
    ]
    
    if not matching_contracts: 
        # No matching contracts found
        return data
    
    # Sort by expiry to find the nearest term
    # Use the first contract's expiryDate (timestamp) as the key for sorting
    matching_contracts.sort(key=lambda x: x['expiryDate'])
    
    if not matching_contracts: return data
    
    # Use the nearest expiry contract for depth analysis
    near_term_symbol = matching_contracts[0]['symbol']
    data["near_term_symbol"] = near_term_symbol
    
    # 2. Get Option Chain Depth (Order Book) for the near-term contract
    option_depth = await fetch_json(session, OPTIONS_DEPTH_URL.format(symbol=near_term_symbol))
    
    if option_depth and 'bids' in option_depth and 'asks' in option_depth:
        # Simplified PCR Logic (for GPT prompt context)
        total_bid_volume = sum([float(b[1]) for b in option_depth['bids']])
        total_ask_volume = sum([float(a[1]) for a in option_depth['asks']])
        
        data["options_liquidity_info"] = {
            "near_term_symbol": near_term_symbol,
            "bid_volume": total_bid_volume,
            "ask_volume": total_ask_volume,
            "bids_count": len(option_depth['bids']),
            "asks_count": len(option_depth['asks'])
        }

    return data


# ---------------- GPT-4o-mini Analysis (Main Logic) ----------------

async def analyze_with_openai(symbol, candles_1h, spot_depth, agg_trades, options_data):
    if not client: return {"side":"none","confidence":0,"reason":"NO_AI_KEY"}
    
    # 1. Prepare Data Summary
    df_1h = pd.DataFrame(candles_1h, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df_1h = df_1h[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    
    # Calculate simple indicators for GPT
    current_price = df_1h['Close'].iloc[-1]
    
    # Simple Candlestick Pattern detection (Last 3 candles)
    last_3_candles = df_1h.tail(3)
    c1, c2, c3 = last_3_candles.iloc[-3], last_3_candles.iloc[-2], last_3_candles.iloc[-1]
    
    pattern_summary = "No clear pattern."
    if c3['Close'] > c3['Open'] and c2['Close'] < c2['Open'] and c1['Close'] < c1['Open'] and c3['Open'] > c2['Close']:
        pattern_summary = "Possible Bullish Engulfing/Morning Star formation."
    elif c3['Close'] < c3['Open'] and c2['Close'] > c2['Open'] and c1['Close'] > c1['Open'] and c3['Open'] < c2['Close']:
        pattern_summary = "Possible Bearish Engulfing/Evening Star formation."
        
    # Spot Market Depth Analysis
    bid_vol_spot = sum([float(b[1]) for b in spot_depth.get('bids', [])])
    ask_vol_spot = sum([float(a[1]) for a in spot_depth.get('asks', [])])
    spot_imbalance = (bid_vol_spot - ask_vol_spot) / (bid_vol_spot + ask_vol_spot) * 100 if (bid_vol_spot + ask_vol_spot) > 0 else 0
    
    # Aggressive Trades Analysis
    buy_vol_agg = sum([float(t['q']) for t in agg_trades if not t['m']])
    sell_vol_agg = sum([float(t['q']) for t in agg_trades if t['m']])
    agg_imbalance = (buy_vol_agg - sell_vol_agg) / (buy_vol_agg + sell_vol_agg) * 100 if (buy_vol_agg + sell_vol_agg) > 0 else 0
    
    # Options Data Summary (for the near-term contract)
    opt_info = options_data.get('options_liquidity_info', {})
    opt_bid_vol = opt_info.get('bid_volume', 0)
    opt_ask_vol = opt_info.get('ask_volume', 0)
    opt_symbol = opt_info.get('near_term_symbol', 'N/A')
    
    # ATR for Risk Management
    highs = df_1h['High'].to_numpy(); lows = df_1h['Low'].to_numpy(); closes = df_1h['Close'].to_numpy()
    current_atr = atr(highs, lows, closes) # using simple ATR (defined below)

    # 2. Construct the Detailed Prompt
    sys_prompt = (
        "You are a sophisticated quantitative crypto analyst using a comprehensive multi-factor model. "
        "Your task is to analyze the provided market data and generate a high-conviction trading signal "
        "(BUY/SELL/HOLD) for the next few hours. Use ALL data points below to formulate your logic. "
        "Strictly adhere to the output format: 'SYMBOL - ACTION - ENTRY:x - SL:y - TP:z - REASON:.. - CONF:n%'"
    )

    usr_prompt = f"""
    ### {symbol} - Advanced Analysis Request
    
    **1. Price Action & Candlesticks (1H TF - Last 720 Candles):**
    - Current Price: {current_price:.2f}
    - Last 3 Candles (OHLCV): \n{last_3_candles.to_string()}
    - Recent Candlestick Pattern: {pattern_summary}
    - **Current ATR (14-period):** {current_atr:.4f} (Use this for SL/TP suggestions: SL ~1.5x ATR, TP ~3x ATR)
    
    **2. Spot Market Flow Analysis (Order Book & Agg Trades):**
    - Spot Depth Imbalance (50 levels): {spot_imbalance:.2f}% (Positive=Buy Pressure, Negative=Sell Pressure)
    - Aggressive Trades Imbalance (Last 100 trades): {agg_imbalance:.2f}% (Positive=Aggressive Buying, Negative=Aggressive Selling)
    
    **3. Options Chain Analysis (Vanilla Options):**
    - Nearest Expiry Contract Symbol: {opt_symbol}
    - Options Bid Volume (Near-Term): {opt_bid_vol:.2f} (Implied Demand/Call side)
    - Options Ask Volume (Near-Term): {opt_ask_vol:.2f} (Implied Supply/Put side)
    - **Interpretation Hint:** High Call liquidity/demand (Bid Vol) suggests bullish expectations; High Put liquidity/supply (Ask Vol) suggests bearish expectations/hedging.
    
    **4. Instructions:**
    - Analyze the long-term trend from the 720 candles.
    - Identify key price action, support, and resistance levels from the historical data.
    - Combine market flow (Spot Depth, Agg Trades) with Options sentiment.
    - **If Confidence is ≥ {int(SIGNAL_CONF_THRESHOLD)}%**, provide a specific **BUY/SELL** signal with calculated ENTRY, SL, and TP.
    - **If Confidence is < {int(SIGNAL_CONF_THRESHOLD)}%**, the action must be **HOLD** or **NONE**.
    """

    try:
        loop = asyncio.get_running_loop()
        def call(): 
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": usr_prompt}
                ],
                max_tokens=600,
                temperature=0.3
            )
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
    signal = {"side": "none", "confidence": 0, "reason": ai_output, "entry": 0, "sl": 0, "tp": 0}
    
    # 1. Parse Confidence
    conf_match = re.search(r'CONF:(\d+)%', ai_output, re.IGNORECASE)
    if conf_match:
        signal['confidence'] = int(conf_match.group(1))

    # 2. Parse Action and T/P (Trade Parameters)
    action_match = re.search(r'(\w+)\s*-\s*(BUY|SELL|HOLD|NONE)', ai_output, re.IGNORECASE)
    entry_match = re.search(r'ENTRY:([\d.]+)', ai_output, re.IGNORECASE)
    sl_match = re.search(r'SL:([\d.]+)', ai_output, re.IGNORECASE)
    tp_match = re.search(r'TP:([\d.]+)', ai_output, re.IGNORECASE)
    reason_match = re.search(r'REASON:(.*?)(\s*-\s*CONF:|\s*$)', ai_output, re.IGNORECASE | re.DOTALL)
    
    if action_match:
        side = action_match.group(2).upper()
        signal['side'] = side
        
        if side in ["BUY", "SELL"]:
            try:
                # Use parsed values if available, otherwise use ATR logic for robustness
                entry = float(entry_match.group(1)) if entry_match else 0
                sl = float(sl_match.group(1)) if sl_match else 0
                tp = float(tp_match.group(1)) if tp_match else 0
                
                # Check for Current Price from the AI's reason field if ENTRY is missing
                if entry == 0: 
                    price_match = re.search(r'Current Price: ([\d.]+)', signal['reason'])
                    if price_match:
                        entry = float(price_match.group(1))
                    else: # Fallback to 0 if Current Price is not found, which will fail the check below
                        entry = 0

                if entry == 0: raise ValueError("Entry price could not be determined.")

                # ATR fallback for SL/TP if parsing fails or values are unreasonable
                if side == "BUY":
                    signal['entry'] = entry
                    signal['sl'] = sl if sl > 0 and sl < entry else entry - (current_atr * 1.5)
                    signal['tp'] = tp if tp > entry else entry + (current_atr * 3.0)
                elif side == "SELL":
                    signal['entry'] = entry
                    signal['sl'] = sl if sl > entry else entry + (current_atr * 1.5)
                    signal['tp'] = tp if tp < entry and tp > 0 else entry - (current_atr * 3.0)
                
            except Exception as e:
                signal['reason'] += f" | (Parsing Error: {e})"
                signal['side'] = "none" # Invalidate signal if essential parameters are missing/bad

    if reason_match:
        signal['reason'] = reason_match.group(1).strip()
        
    return signal

# ---------------- Professional Candlestick Chart (with Flying Raijin Logo) ----------------

def plot_signal_chart(symbol, candles, signal):
    
    # 1. Prepare Data for mplfinance (Pandas DataFrame)
    df_candles = pd.DataFrame(candles, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
    df_candles['OpenTime'] = pd.to_datetime(df_candles['OpenTime'], unit='ms')
    df_candles = df_candles.set_index(pd.DatetimeIndex(df_candles['OpenTime']))
    df_candles = df_candles[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    # Add SMA 200 for long-term trend
    df_candles['SMA200'] = df_candles['Close'].rolling(window=200).mean()

    apds = [
        mpf.make_addplot(df_candles['SMA200'], color='#FFA500', panel=0, width=1.0, secondary_y=False), 
    ]

    # 3. Prepare horizontal lines for signals
    hlines = []
    colors = []; linestyles = []
    
    if signal["side"] in ["BUY", "SELL"]:
        hlines.extend([signal["entry"], signal["sl"], signal["tp"]])
        colors = ['blue', 'red', 'green']
        linestyles = ['-', '--', '--']
        
    # 4. Define the chart style 
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo', 
        gridcolor='#e0e0e0',    
        facecolor='#ffffff',    
        figcolor='#ffffff',     
        y_on_right=False,       
        marketcolors=mpf.make_marketcolors(
            up='g',             
            down='r',           
            edge='inherit',     
            wick='inherit',     
            volume='inherit'    
        )
    )

    # 5. Plot the chart
    fig, axlist = mpf.plot(
        df_candles.iloc[-300:], # Plotting last 300 candles for better visual
        type='candle',
        style=s,
        title=f"{symbol} 1H Advanced Signal ({signal.get('side','?')}) | Conf {signal.get('confidence',0)}%",
        ylabel='Price',
        addplot=apds,           
        figscale=1.5,           
        returnfig=True,         
        hlines=dict(hlines=hlines, colors=colors, linestyles=linestyles, linewidths=1.5, alpha=0.9),
    )

    # 6. Add 'Flying Raijin' Logo/Watermark
    ax = axlist[0] # Get the main candlestick axes
    logo_text = "⚡ Flying Raijin ⚡"
    
    ax.text(
        0.99, 0.01,             # Position: Bottom Right
        logo_text,              
        transform=ax.transAxes, 
        fontsize=14,
        fontweight='bold',
        color='#FFD700',        # Gold color for striking effect
        ha='right',             
        va='bottom',            
        alpha=0.8,
        bbox=dict(facecolor='#333333', alpha=0.7, edgecolor='#FFD700', linewidth=1, boxstyle='round,pad=0.5')
    )

    # 7. Save the chart
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches='tight')
    plt.close(fig)
    return tmp.name

# ---------------- Fetch & Telegram (Keep as is) ----------------
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
        
    init_redis_plain() # Initialize Redis at the start
    
    async with aiohttp.ClientSession() as session:
        startup=f"🤖 Advanced BTC/ETH Bot Started (Options, 720H Candles, GPT-4o-mini) • Poll {POLL_INTERVAL//60}min"
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
                tasks[f"{sym}_options"] = get_options_data_for_symbol(session, sym) # Fetch Options data

            
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
                    c1h = data.get("1h"); spot_depth = data.get("depth"); agg_data = data.get("aggTrades"); options_data = data.get("options")

                    if not all([c1h, spot_depth, agg_data, options_data]):
                        print(f"{sym}: Missing critical data (1H Candle/Spot Depth/AggTrades/Options), skipping"); continue

                    # 1. Run Advanced AI Analysis
                    final_signal = await analyze_with_openai(sym, c1h, spot_depth, agg_data, options_data)
                    
                    if final_signal["side"] in ["BUY", "SELL"]:
                        
                        if final_signal["confidence"] >= SIGNAL_CONF_THRESHOLD:
                            store_signal(sym, final_signal)

                            msg=f"**🔥 Advanced AI Signal ({final_signal['confidence']}%)**\n\n**Asset:** {sym}\n**Action:** {final_signal['side']}\n**Entry:** `{fmt_price(final_signal['entry'])}`\n**SL:** `{fmt_price(final_signal['sl'])}` | **TP:** `{fmt_price(final_signal['tp'])}`\n\n**AI Logic:** {final_signal['reason']}"
                            
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
