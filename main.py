#!/usr/bin/env python3
# main.py - Crypto bot v5.2 (Redis integrated - non-TLS friendly)
# - Uses Redis host/port (non-TLS) from env
# - Stores LTP and candles streams
# - Keeps original signal/chart logic and Telegram/OpenAI integration

import os, json, asyncio, traceback, time
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from tempfile import NamedTemporaryFile

# plotting (server-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

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

POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", 1800)))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&limit=100"
TICKER_24H_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"

# ---------------- Redis init (non-TLS plain) ----------------
REDIS = None
def init_redis_plain():
    global REDIS
    host = os.getenv("REDIS_HOST")
    port = int(os.getenv("REDIS_PORT")) if os.getenv("REDIS_PORT") else None
    user = os.getenv("REDIS_USER") or None
    password = os.getenv("REDIS_PASSWORD") or None
    # allow REDIS_URL fallback (non-TLS redis://)
    url = os.getenv("REDIS_URL")

    # Try host/port first (works with free Redis)
    if host and port:
        try:
            print("Connecting to Redis (plain host/port)...")
            REDIS = redis.Redis(host=host, port=port, username=user, password=password, decode_responses=True)
            print("Redis ping:", REDIS.ping())
            return
        except Exception as e:
            print("Plain host/port Redis failed:", e)
            traceback.print_exc()
            REDIS = None

    # Fallback to redis.from_url (non-TLS redis://)
    if url:
        try:
            print("Trying redis.from_url() fallback...")
            REDIS = redis.Redis.from_url(url, decode_responses=True)
            print("Redis ping (from_url):", REDIS.ping())
            return
        except Exception as e:
            print("redis.from_url() failed:", e)
            traceback.print_exc()
            REDIS = None

    print("Redis not available; continuing without Redis.")

init_redis_plain()

# ---------------- Redis helpers ----------------
def safe_call(fn, *args, **kwargs):
    try:
        if REDIS:
            return fn(*args, **kwargs)
    except Exception as e:
        print("Redis error:", e)
    return None

def safe_setex(key, ttl, val):
    return safe_call(REDIS.setex, key, ttl, val) if REDIS else None

def safe_hset(h, field, val):
    return safe_call(REDIS.hset, h, field, val) if REDIS else None

def safe_xadd(key, mapping, maxlen=None):
    if not REDIS: return None
    try:
        if maxlen:
            return REDIS.xadd(key, mapping=mapping, maxlen=maxlen, approximate=True)
        else:
            return REDIS.xadd(key, mapping=mapping)
    except Exception as e:
        print("Redis xadd error:", e)
        return None

# Retention lengths (entries) - adjust as needed
STREAM_MAXLEN = {"30m":2880, "1h":2160, "4h":1080, "1d":730}

def store_ltp(symbol, price, ts):
    safe_setex(f"crypto:ltp:{symbol}", 30, f"{price},{ts}")

def store_24h(symbol, stats):
    if REDIS:
        try:
            REDIS.setex(f"crypto:24h:{symbol}", 300, json.dumps(stats))
        except Exception as e:
            print("store_24h error:", e)

def store_candle(symbol, tf, candle):
    key = f"candles:{tf}:{symbol}"
    safe_xadd(key, mapping=candle, maxlen=STREAM_MAXLEN.get(tf))

def store_signal(symbol, signal):
    safe_hset("signals:active", symbol, json.dumps(signal))

def store_prediction(symbol, pred):
    if REDIS:
        try:
            REDIS.setex(f"predictions:{symbol}", 3600, json.dumps(pred))
        except Exception as e:
            print("store_prediction error:", e)

# ---------------- Utils & indicators ----------------
def fmt_price(p): return f"{p:.6f}" if abs(p)<1 else f"{p:.2f}"

def ema(values, period):
    if len(values)<period: return []
    k = 2/(period+1); prev = sum(values[:period])/period
    arr = [None]*(period-1)+[prev]
    for v in values[period:]:
        prev = v*k + prev*(1-k); arr.append(prev)
    return arr

def horizontal_levels(closes, highs, lows, lookback=50, binsize=0.002):
    pts = closes[-lookback:]+highs[-lookback:]+lows[-lookback:]
    lvls=[] 
    for p in pts:
        found=False
        for lv in lvls:
            if abs((lv["price"]-p)/p) < binsize:
                lv["count"]+=1
                lv["price"]=(lv["price"]*(lv["count"]-1)+p)/lv["count"]
                found=True; break
        if not found: lvls.append({"price":p,"count":1})
    lvls.sort(key=lambda x:-x["count"])
    return [lv["price"] for lv in lvls[:5]]

def analyze_trade_logic(candles, rr_min=1.5):
    closes=[c[3] for c in candles]; highs=[c[1] for c in candles]; lows=[c[2] for c in candles]
    if len(closes)<30: return {"side":"none","confidence":0,"reason":"not enough data"}
    es,el = ema(closes,9)[-1], ema(closes,21)[-1]; price = closes[-1]; lvls = horizontal_levels(closes,highs,lows)
    sup = max([lv for lv in lvls if lv<price], default=None); res = min([lv for lv in lvls if lv>price], default=None)
    conf=50; reasons=[]
    if price<es and price<el: conf+=10; reasons.append("below EMAs")
    elif price>es and price>el: conf+=10; reasons.append("above EMAs")
    else: conf-=5; reasons.append("EMAs mixed")
    if price<es and res:
        entry, stop = price, res*1.003; tgt = sup if sup else price-(stop-entry)*1.5
        rr = (entry-tgt)/(stop-entry) if stop>entry else 0
        if rr>=rr_min: return {"side":"SELL","entry":entry,"sl":stop,"tp":tgt,"confidence":conf+10,"reason":"; ".join(reasons)}
    if price>es and res:
        entry, stop = price, (sup*0.997 if sup else price*0.99); tgt = res
        rr = (tgt-entry)/(entry-stop) if entry>stop else 0
        if rr>=rr_min: return {"side":"BUY","entry":entry,"sl":stop,"tp":tgt,"confidence":conf+10,"reason":"; ".join(reasons)}
    return {"side":"none","confidence":conf,"reason":"; ".join(reasons)}

# ---------------- Chart ----------------
def plot_signal_chart(symbol, candles, signal):
    dates=[datetime.utcfromtimestamp(int(x[0])/1000) for x in candles]
    closes=[float(x[4]) for x in candles]; highs=[float(x[2]) for x in candles]; lows=[float(x[3]) for x in candles]
    x=date2num(dates); fig,ax=plt.subplots(figsize=(12,6))
    for xi,c in zip(x,candles):
        o,h,l,cl=float(c[1]),float(c[2]),float(c[3]),float(c[4])
        color="g" if cl>=o else "r"; ax.plot([xi,xi],[l,h],color=color); ax.plot(xi,cl,"o",color=color)
    try:
        ax.axhline(signal["entry"],color="blue",ls="--",label=f"Entry {fmt_price(signal['entry'])}")
        ax.axhline(signal["sl"],color="red",ls="--",label=f"SL {fmt_price(signal['sl'])}")
        ax.axhline(signal["tp"],color="green",ls="--",label=f"TP {fmt_price(signal['tp'])}")
    except Exception:
        pass
    ax.legend(); ax.set_title(f"{symbol} Signal {signal.get('side','?')} | Conf {signal.get('confidence',0)}%")
    tmp=NamedTemporaryFile(delete=False,suffix=".png"); fig.savefig(tmp.name); plt.close(fig); return tmp.name

# ---------------- Fetch & AI ----------------
async def fetch_json(session,url):
    try:
        async with session.get(url,timeout=20) as r:
            if r.status!=200: return None
            return await r.json()
    except Exception as e:
        print("fetch_json error:", e)
        return None

async def ask_openai_for_signals(summary):
    if not client: return None
    sys="You are pro crypto trader. Only output signals."
    usr=f"{summary}\nRules: Conf≥{int(SIGNAL_CONF_THRESHOLD)}% Format: SYMBOL - BUY/SELL - ENTRY:x - SL:y - TP:z - REASON:.. - CONF:n%"
    try:
        loop=asyncio.get_running_loop()
        def call(): return client.chat.completions.create(model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
            max_tokens=400,temperature=0.0)
        resp=await loop.run_in_executor(None,call)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("ask_openai_for_signals error:", e)
        return None

# ---------------- Telegram ----------------
async def send_text(session,text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        await session.post(url,json={"chat_id":TELEGRAM_CHAT_ID,"text":text})
    except Exception as e:
        print("send_text error:", e)

async def send_photo(session,caption,path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(caption); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path,"rb") as f:
            data=aiohttp.FormData(); data.add_field("chat_id",TELEGRAM_CHAT_ID); data.add_field("caption",caption); data.add_field("photo",f)
            await session.post(url,data=data)
    except Exception as e:
        print("send_photo error:", e)

# ---------------- Main loop ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession() as session:
        startup=f"🤖 Bot started • {len(SYMBOLS)} symbols • Poll {POLL_INTERVAL}s"
        print(startup); await send_text(session,startup)
        it=0
        while True:
            it+=1; print(f"\nITER {it} @ {datetime.now().strftime('%H:%M:%S')}")
            for sym in SYMBOLS:
                try:
                    c30 = await fetch_json(session,CANDLE_URL.format(symbol=sym,tf="30m"))
                    c1h = await fetch_json(session,CANDLE_URL.format(symbol=sym,tf="1h"))
                    c4h = await fetch_json(session,CANDLE_URL.format(symbol=sym,tf="4h"))
                    c1d = await fetch_json(session,CANDLE_URL.format(symbol=sym,tf="1d"))
                    stats24h = await fetch_json(session,TICKER_24H_URL.format(symbol=sym))
                    if not c30 or not c1h:
                        print(f"{sym}: missing data, skipping"); continue

                    can30=[[float(x[1]),float(x[2]),float(x[3]),float(x[4])] for x in c30]
                    can1h=[[float(x[1]),float(x[2]),float(x[3]),float(x[4])] for x in c1h]

                    # Redis stores (if available)
                    try:
                        store_ltp(sym, can30[-1][3], int(datetime.now().timestamp()))
                        if stats24h: store_24h(sym, stats24h)
                    except Exception as e:
                        print("Redis store error:", e)

                    for tf,data in [("30m",c30),("1h",c1h),("4h",c4h),("1d",c1d)]:
                        if not data: continue
                        last=data[-1]
                        candle_data={"t":str(int(last[0])),"o":str(last[1]),"h":str(last[2]),
                                     "l":str(last[3]),"c":str(last[4]),"v":str(last[5])}
                        store_candle(sym, tf, candle_data)

                    local=analyze_trade_logic(can30)
                    ai=await ask_openai_for_signals(json.dumps({"symbol":sym,"last_price":can30[-1][3]},indent=2))

                    if local["side"]!="none":
                        conf=local["confidence"]+(10 if ai and "NO_SIGNAL" not in ai else 0)
                        signal={"side":local["side"],"entry":local["entry"],"sl":local["sl"],
                                "tp":local["tp"],"confidence":conf,"reason":local["reason"]}
                        try: store_signal(sym, signal)
                        except Exception as e: print("store_signal error:", e)

                        msg=f"{sym} {signal['side']} | Entry {fmt_price(signal['entry'])} | SL {fmt_price(signal['sl'])} | TP {fmt_price(signal['tp'])} | Conf {signal['confidence']}%\\nReason: {signal['reason']}"
                        chart=plot_signal_chart(sym,c30,signal)
                        await send_photo(session,msg,chart)
                        print("⚡",msg)
                    else:
                        print(f"{sym}: no signal")
                except Exception as e:
                    print("Error",sym,e)
            await asyncio.sleep(POLL_INTERVAL)

if __name__=="__main__":
    try: asyncio.run(enhanced_loop())
    except KeyboardInterrupt: print("Stopped by user")
