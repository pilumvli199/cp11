#!/usr/bin/env python3
# updated_main.py - Professional Crypto Bot (MTF, ATR, Depth, AggTrades, Candlestick Chart)

import os, json, asyncio, traceback, time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from tempfile import NamedTemporaryFile

# plotting (server-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from mplfinance.original_flavor import candlestick_ohlc # For professional charts

# OpenAI client (optional)
from openai import OpenAI

# Redis (non-TLS)
import redis

load_dotenv()

# ---------------- CONFIG ----------------
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]

POLL_INTERVAL = max(15, int(os.getenv("POLL_INTERVAL", 60))) # Polling every 60s
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 75.0))

# API Endpoints
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&limit={limit}"
TICKER_24H_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
DEPTH_URL = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit=50" # Top 50 Bids/Asks
AGGT_URL = "https://api.binance.com/api/v3/aggTrades?symbol={symbol}&limit=100" # Last 100 Trades

CANDLE_LIMITS = {"30m": 100, "1h": 100, "4h": 100, "1d": 200}
TIME_FRAMES = ["30m", "1h", "4h", "1d"]
RISK_REWARD_RATIO = 1.5

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------- Redis init (non-TLS plain) ----------------
REDIS = None
def init_redis_plain():
    global REDIS
    # ... (Redis connection logic - unchanged)
    host = os.getenv("REDIS_HOST")
    port = int(os.getenv("REDIS_PORT")) if os.getenv("REDIS_PORT") else None
    user = os.getenv("REDIS_USER") or None
    password = os.getenv("REDIS_PASSWORD") or None
    url = os.getenv("REDIS_URL")

    if host and port:
        try:
            print("Connecting to Redis (plain host/port)...")
            REDIS = redis.Redis(host=host, port=port, username=user, password=password, decode_responses=True)
            print("Redis ping:", REDIS.ping())
            return
        except Exception: REDIS = None
    if url and not REDIS:
        try:
            print("Trying redis.from_url() fallback...")
            REDIS = redis.Redis.from_url(url, decode_responses=True)
            print("Redis ping (from_url):", REDIS.ping())
            return
        except Exception: REDIS = None
    print("Redis not available; continuing without Redis.")

init_redis_plain()

# ---------------- Redis helpers (Updated for Depth/AggTrades) ----------------
def safe_call(fn, *args, **kwargs):
    try:
        if REDIS: return fn(*args, **kwargs)
    except Exception as e: print("Redis error:", e)
    return None

def safe_setex(key, ttl, val):
    return safe_call(REDIS.setex, key, ttl, val) if REDIS else None

def safe_hset(h, field, val):
    return safe_call(REDIS.hset, h, field, val) if REDIS else None

def safe_xadd(key, mapping, maxlen=None):
    if not REDIS: return None
    try:
        if maxlen: return REDIS.xadd(key, mapping, maxlen=maxlen, approximate=True)
        else: return REDIS.xadd(key, mapping)
    except Exception as e:
        print("Redis xadd error:", e)
        return None

STREAM_MAXLEN = {"30m":2880, "1h":2160, "4h":1080, "1d":730}

def store_ltp(symbol, price, ts): safe_setex(f"crypto:ltp:{symbol}", 30, f"{price},{ts}")
def store_24h(symbol, stats):
    if REDIS:
        try: REDIS.setex(f"crypto:24h:{symbol}", 300, json.dumps(stats))
        except Exception as e: print("store_24h error:", e)

# NEW: Store Market Depth Data
def store_depth(symbol, depth_data):
    # Store with short TTL as depth changes fast
    if REDIS:
        try: REDIS.setex(f"crypto:depth:{symbol}", 15, json.dumps(depth_data))
        except Exception as e: print("store_depth error:", e)

# NEW: Store Aggregated Trades Data (as a temporary set)
def store_aggtrades(symbol, agg_data):
    if REDIS:
        try: REDIS.setex(f"crypto:aggtrades:{symbol}", 15, json.dumps(agg_data))
        except Exception as e: print("store_aggtrades error:", e)

def store_candle(symbol, tf, candle):
    key = f"candles:{tf}:{symbol}"
    safe_xadd(key, candle, maxlen=STREAM_MAXLEN.get(tf))

def store_signal(symbol, signal): safe_hset("signals:active", symbol, json.dumps(signal))

# ---------------- Utils & indicators (Unchanged) ----------------
def fmt_price(p): return f"{p:.6f}" if abs(p)<1 else f"{p:.2f}"

def ema(values, period):
    if len(values) < period: return [None] * len(values)
    k = 2 / (period + 1); arr = [None] * (period - 1)
    prev = np.mean(values[:period]); arr.append(prev)
    for v in values[period:]: prev = v * k + prev * (1 - k); arr.append(prev)
    return arr

def atr(highs, lows, closes, period=14):
    if len(closes) < period: return 0.0
    tr = []
    for i in range(1, len(closes)):
        tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    initial_atr = np.mean(tr[:period])
    atrs = [None] * period + [initial_atr]
    for i in range(period, len(tr)):
        atrs.append((atrs[-1] * (period - 1) + tr[i]) / period)
    return atrs[-1] if atrs else 0.0

def rsi(closes, period=14):
    if len(closes) < period + 1: return None
    deltas = np.diff(closes)
    seed = deltas[:period]; up = seed[seed >= 0].sum() / period; down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else (100 if up > 0 else 0)
    rsi_vals = [100. - 100. / (1. + rs)]
    for i in range(period, len(closes) - 1):
        delta = deltas[i]; upval = delta if delta >= 0 else 0; downval = -delta if delta < 0 else 0
        up = (up * (period - 1) + upval) / period; down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else (100 if up > 0 else 0)
        rsi_vals.append(100. - 100. / (1. + rs))
    return rsi_vals[-1] if rsi_vals else None

# ---------------- New Analysis Logic ----------------

def analyze_depth_pressure(depth_data):
    if not depth_data or 'bids' not in depth_data or 'asks' not in depth_data: return 0, 0
    
    # Calculate cumulative volume for top N bids and asks
    bid_volume = sum([float(b[1]) for b in depth_data['bids']])
    ask_volume = sum([float(a[1]) for a in depth_data['asks']])
    
    total_volume = bid_volume + ask_volume
    if total_volume == 0: return 0, 0
    
    # Pressure is the ratio of Bid Volume to Total Volume
    # > 0.5 means more buying pressure, < 0.5 means more selling pressure
    pressure_ratio = bid_volume / total_volume 
    
    # Calculate imbalance: how much bigger is one side than the other (percentage difference)
    imbalance_pct = (bid_volume - ask_volume) / total_volume * 100
    
    return pressure_ratio, imbalance_pct

def analyze_aggtrades_momentum(agg_data):
    if not agg_data: return 0, 0
    
    buy_vol = 0
    sell_vol = 0
    
    # AggTrades: 'm' is True if the buyer is the market maker (i.e., it was a SELL trade)
    # 'M' is False if the seller is the market maker (i.e., it was a BUY trade)
    for trade in agg_data:
        qty = float(trade['q'])
        is_sell_trade = trade['m'] # True if it was a sell trade (Taker is buyer)
        
        if not is_sell_trade: # Taker is BUYER (Aggressive Buy)
            buy_vol += qty
        else: # Taker is SELLER (Aggressive Sell)
            sell_vol += qty
            
    total_vol = buy_vol + sell_vol
    if total_vol == 0: return 0, 0
    
    # Momentum is the ratio of Aggressive Buy Volume to Total Aggressive Volume
    momentum_ratio = buy_vol / total_vol
    
    # Imbalance: how much aggressive buying vs selling
    imbalance_pct = (buy_vol - sell_vol) / total_vol * 100
    
    return momentum_ratio, imbalance_pct

def analyze_trade_logic(candles_30m, candles_1h, candles_4h, depth_data, agg_data, rr_min=RISK_REWARD_RATIO):
    # (Existing Kline Data Extraction & Indicator Calculation - Unchanged)
    closes = np.array([float(c[4]) for c in candles_30m]); highs = np.array([float(c[2]) for c in candles_30m]); lows = np.array([float(c[3]) for c in candles_30m]); volumes = np.array([float(c[5]) for c in candles_30m])
    if len(closes) < 30: return {"side":"none","confidence":0,"reason":"Not enough 30m data"}
    
    price = closes[-1]; es_9, es_21 = ema(closes, 9)[-1], ema(closes, 21)[-1]; current_rsi = rsi(closes); current_atr = atr(highs, lows, closes)
    
    h_closes = np.array([float(c[4]) for c in candles_1h]); d_closes = np.array([float(c[4]) for c in candles_4h])
    h_ema_50 = ema(h_closes, 50)[-1] if len(h_closes) >= 50 else price; d_ema_50 = ema(d_closes, 50)[-1] if len(d_closes) >= 50 else price
    trend = "sideways"
    if price > h_ema_50 and price > d_ema_50: trend = "up"
    elif price < h_ema_50 and price < d_ema_50: trend = "down"
    
    conf = 50; reasons = []
    side = "none"
    
    # NEW: Order Book & AggTrades Analysis
    depth_ratio, depth_imbalance = analyze_depth_pressure(depth_data)
    agg_ratio, agg_imbalance = analyze_aggtrades_momentum(agg_data)

    # --- BUY Signal Logic ---
    if price > es_9 and es_9 > es_21 and trend == "up":
        if current_rsi and 40 <= current_rsi <= 70:
            
            # ATR-based SL/TP
            sl_dist = current_atr * 1.5; tp_dist = current_atr * (1.5 * rr_min)
            entry, stop = price, price - sl_dist
            tgt = price + tp_dist
            
            # Additional Confirmation Checks (Confidence Boost)
            if depth_imbalance > 10: # More than 10% more Bid volume
                conf += 10; reasons.append(f"Strong Bid Pressure ({depth_imbalance:.1f}%)")
            
            if agg_imbalance > 15: # More than 15% Aggressive Buy volume
                conf += 10; reasons.append(f"Aggressive Buying Momentum ({agg_imbalance:.1f}%)")
                
            conf += 15; reasons.append(f"Uptrend confirmed by {trend} MTF, EMAs crossed up, RSI is {current_rsi:.2f}")
            side = "BUY"

    # --- SELL Signal Logic ---
    elif price < es_9 and es_9 < es_21 and trend == "down":
        if current_rsi and 30 <= current_rsi <= 60:
            
            # ATR-based SL/TP
            sl_dist = current_atr * 1.5; tp_dist = current_atr * (1.5 * rr_min)
            entry, stop = price, price + sl_dist
            tgt = price - tp_dist
            
            # Additional Confirmation Checks (Confidence Boost)
            if depth_imbalance < -10: # More than 10% more Ask volume (selling)
                conf += 10; reasons.append(f"Strong Ask Pressure ({abs(depth_imbalance):.1f}%)")
            
            if agg_imbalance < -15: # More than 15% Aggressive Sell volume
                conf += 10; reasons.append(f"Aggressive Selling Momentum ({abs(agg_imbalance):.1f}%)")
                
            conf += 15; reasons.append(f"Downtrend confirmed by {trend} MTF, EMAs crossed down, RSI is {current_rsi:.2f}")
            side = "SELL"
    
    if side != "none":
        return {"side": side, "entry": entry, "sl": stop, "tp": tgt, "confidence": conf, "reason": "; ".join(reasons)}
        
    return {"side":"none","confidence":conf,"reason":"No strong setup. MTF trend: "+trend}

# ---------------- Professional Candlestick Chart ----------------
def plot_signal_chart(symbol, candles, signal):
    dates = [datetime.utcfromtimestamp(int(x[0])/1000) for x in candles]
    
    # Convert data to OHLC format for mplfinance: (date_num, open, high, low, close)
    ohlc = []
    for i, c in enumerate(candles):
        # [0] open time, [1] open, [2] high, [3] low, [4] close, [5] volume
        ohlc.append([date2num(dates[i]), float(c[1]), float(c[2]), float(c[3]), float(c[4])])
    
    # Prepare EMAs for plotting
    closes = np.array([float(x[4]) for x in candles])
    ema_9 = ema(closes, 9)
    ema_21 = ema(closes, 21)
    
    # Create subplots
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.grid(True)
    ax.set_facecolor('#f0f0f0') 
    
    # 1. Plot Candlesticks
    candlestick_ohlc(ax, ohlc, width=0.0008, colorup='g', colordown='r', alpha=0.9)
    
    # 2. Plot EMAs
    ax.plot(date2num(dates), ema_9, label='EMA 9', color='#FFC300', linewidth=1.5) # Yellow/Gold
    ax.plot(date2num(dates), ema_21, label='EMA 21', color='#DAF7A6', linewidth=1.5) # Light Green

    # 3. Plot Signal Levels (Horizontal Lines)
    try:
        # Use price formatting for labels
        ax.axhline(signal["entry"], color="blue", ls="-", linewidth=2, label=f"Entry: {fmt_price(signal['entry'])}")
        ax.axhline(signal["sl"], color="red", ls="--", linewidth=2, label=f"SL: {fmt_price(signal['sl'])}")
        ax.axhline(signal["tp"], color="green", ls="--", linewidth=2, label=f"TP: {fmt_price(signal['tp'])}")
    except Exception:
        pass
        
    # Format the X-axis (dates)
    ax.xaxis_date(); ax.autoscale_view()
    ax.set_title(f"{symbol} 30M Signal ({signal.get('side','?')}) | Conf {signal.get('confidence',0)}% - MTF, ATR, Depth Confirmed", fontsize=16)
    ax.legend(loc='best', fontsize=10)
    
    # Save the chart
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches='tight')
    plt.close(fig)
    return tmp.name

# ---------------- Fetch & AI (Unchanged) ----------------
async def fetch_json(session,url):
    try:
        async with session.get(url,timeout=20) as r:
            if r.status!=200: return None
            return await r.json()
    except Exception as e:
        print("fetch_json error:", e)
        return None

async def ask_openai_for_signals(symbol, latest_30m, summary):
    if not client: return "NO_AI_KEY"
    sys="You are a professional crypto trading analyst. Provide signals only if you are highly confident based on the provided data and analysis. Only output signals."
    last_5_candles = latest_30m[-5:]
    usr=f"SYMBOL: {symbol}\n30M Analysis Summary: {summary}\nLast 5 30M Candles (OHLCV): {last_5_candles}\n\nRules: Only generate signal if Conf≥{int(SIGNAL_CONF_THRESHOLD)}%. Format: SYMBOL - BUY/SELL - ENTRY:x - SL:y - TP:z - REASON:.. - CONF:n%"
    try:
        loop=asyncio.get_running_loop()
        def call(): return client.chat.completions.create(model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
            max_tokens=400,temperature=0.2)
        resp=await loop.run_in_executor(None,call)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("ask_openai_for_signals error:", e)
        return "NO_AI_RESPONSE_ERROR"

# ---------------- Telegram (Unchanged) ----------------
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

# ---------------- Main loop (Updated Fetching) ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession() as session:
        startup=f"🤖 Professional Bot Started (MTF, ATR, Depth, AggTrades) • {len(SYMBOLS)} symbols • Poll {POLL_INTERVAL}s"
        print(startup); await send_text(session,startup)
        it=0
        while True:
            it+=1; print(f"\nITER {it} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # --- Fetch all market data concurrently ---
            tasks = {}
            for sym in SYMBOLS:
                tasks[f"{sym}_24h"] = fetch_json(session, TICKER_24H_URL.format(symbol=sym))
                tasks[f"{sym}_depth"] = fetch_json(session, DEPTH_URL.format(symbol=sym)) # NEW
                tasks[f"{sym}_aggTrades"] = fetch_json(session, AGGT_URL.format(symbol=sym)) # NEW
                for tf in TIME_FRAMES:
                    limit = CANDLE_LIMITS[tf]
                    tasks[f"{sym}_{tf}"] = fetch_json(session, CANDLE_URL.format(symbol=sym, tf=tf, limit=limit))
            
            results = await asyncio.gather(*tasks.values())
            
            fetched_data = {}; start_time = time.time()
            for key, res in zip(tasks.keys(), results):
                sym, tf = key.split('_', 1)
                if sym not in fetched_data: fetched_data[sym] = {}
                fetched_data[sym][tf] = res

            # --- Process each symbol ---
            for sym in SYMBOLS:
                try:
                    data = fetched_data.get(sym, {})
                    c30 = data.get("30m"); c1h = data.get("1h"); c4h = data.get("4h"); stats24h = data.get("24h")
                    depth_data = data.get("depth"); agg_data = data.get("aggTrades")

                    if not all([c30, c1h, c4h, depth_data, agg_data]):
                        print(f"{sym}: missing critical data (Kline/Depth/AggTrades), skipping"); continue

                    # Store data to Redis
                    store_ltp(sym, float(c30[-1][4]), int(datetime.now().timestamp()));
                    if stats24h: store_24h(sym, stats24h);
                    store_depth(sym, depth_data); # Store Depth
                    store_aggtrades(sym, agg_data); # Store AggTrades

                    for tf, candles in [("30m", c30), ("1h", c1h), ("4h", c4h), ("1d", data.get("1d"))]:
                        if candles:
                            last=candles[-1]
                            candle_data={"t":str(int(last[0])),"o":str(last[1]),"h":str(last[2]),
                                         "l":str(last[3]),"c":str(last[4]),"v":str(last[5])}
                            store_candle(sym, tf, candle_data)

                    # Run Enhanced Logic with new data points
                    local_signal = analyze_trade_logic(c30, c1h, c4h, depth_data, agg_data)
                    
                    # Ask AI for additional insight
                    ai_prediction = await ask_openai_for_signals(sym, c30, local_signal["reason"])
                    
                    final_signal = local_signal
                    
                    if final_signal["side"] != "none":
                        # Confidence Boost if AI supports the signal
                        if final_signal["side"] in ai_prediction:
                             final_signal["confidence"] = min(99, final_signal["confidence"] + 5)
                             
                        if final_signal["confidence"] >= SIGNAL_CONF_THRESHOLD:
                            try: store_signal(sym, final_signal)
                            except Exception as e: print("store_signal error:", e)

                            msg=f"**🔥 High Confidence Signal ({final_signal['confidence']}%)**\n\n**Asset:** {sym}\n**Action:** {final_signal['side']}\n**Entry:** `{fmt_price(final_signal['entry'])}`\n**SL:** `{fmt_price(final_signal['sl'])}` | **TP:** `{fmt_price(final_signal['tp'])}`\n\n**Logic:** {final_signal['reason']}"
                            chart=plot_signal_chart(sym, c30, final_signal)
                            await send_photo(session,msg,chart)
                            print("⚡",msg.replace('\n', ' '))
                        else:
                             print(f"{sym}: signal generated but confidence too low ({final_signal['confidence']:.0f}%)")
                    else:
                        print(f"{sym}: no signal. AI: {ai_prediction[:30]}...")
                        
                except Exception as e:
                    print(f"Error processing {sym}: {e}")
                    traceback.print_exc()
                    
            print(f"Processing time: {time.time() - start_time:.2f}s")
            await asyncio.sleep(POLL_INTERVAL)

if __name__=="__main__":
    try: asyncio.run(enhanced_loop())
    except KeyboardInterrupt: print("Stopped by user")
