# main.py - Enhanced with GPT-driven Entry/SL/TP
import os
import asyncio
import aiohttp
import traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple

load_dotenv()

# --- Config ---
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]
POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", 1800)))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 75.0))

RSI_PERIOD = 14
MA_SHORT = 7
MA_LONG = 21
VOLUME_MULTIPLIER = 1.5
MIN_CANDLES_FOR_ANALYSIS = 50
LOOKBACK_PERIOD = 48

price_history = {}
signal_history = []

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=100"
ORDER_BOOK_URL = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit=10"


# ---------------- Indicators ----------------
def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0: gains.append(change); losses.append(0)
        else: gains.append(0); losses.append(abs(change))
    if len(gains) < period: return None
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def calculate_moving_averages(prices: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(prices) < MA_LONG: return None, None
    ma_short = sum(prices[-MA_SHORT:]) / MA_SHORT if len(prices) >= MA_SHORT else None
    ma_long = sum(prices[-MA_LONG:]) / MA_LONG
    return ma_short, ma_long

def detect_volume_spike(volumes: List[float]) -> bool:
    if len(volumes) < 10: return False
    avg_volume = sum(volumes[-10:-1]) / 9
    return volumes[-1] > (avg_volume * VOLUME_MULTIPLIER)

def enhanced_levels(candles, lookback=LOOKBACK_PERIOD):
    if not candles or len(candles) < 10: return (None, None, None, None, None)
    arr = candles[-lookback:] if len(candles) >= lookback else candles
    highs = [c[1] for c in arr]; lows = [c[2] for c in arr]; closes = [c[3] for c in arr]
    highs_sorted = sorted(highs, reverse=True); lows_sorted = sorted(lows)
    primary_resistance = sum(highs_sorted[:3]) / 3 if len(highs_sorted) >= 3 else None
    primary_support = sum(lows_sorted[:3]) / 3 if len(lows_sorted) >= 3 else None
    secondary_resistance = sum(highs_sorted[3:6]) / 3 if len(highs_sorted) >= 6 else None
    secondary_support = sum(lows_sorted[3:6]) / 3 if len(lows_sorted) >= 6 else None
    mid_level = (primary_resistance + primary_support) / 2 if primary_resistance and primary_support else closes[-1]
    return primary_support, primary_resistance, secondary_support, secondary_resistance, mid_level


# ---------------- Fetching ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=15) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception:
        return None

async def fetch_enhanced_data(session, symbol):
    ticker, candles, orderbook = await asyncio.gather(
        fetch_json(session, TICKER_URL.format(symbol=symbol)),
        fetch_json(session, CANDLE_URL.format(symbol=symbol)),
        fetch_json(session, ORDER_BOOK_URL.format(symbol=symbol))
    )
    out = {}
    if ticker:
        out["price"] = float(ticker.get("lastPrice", 0))
        out["volume"] = float(ticker.get("volume", 0))
    if isinstance(candles, list) and len(candles) >= MIN_CANDLES_FOR_ANALYSIS:
        out["candles"] = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in candles]
        out["times"] = [int(x[0]) // 1000 for x in candles]
        out["volumes"] = [float(x[5]) for x in candles]
        closes = [float(x[4]) for x in candles]
        out["rsi"] = calculate_rsi(closes, RSI_PERIOD)
        out["ma_short"], out["ma_long"] = calculate_moving_averages(closes)
        out["volume_spike"] = detect_volume_spike(out["volumes"])
    return out


# ---------------- OpenAI Analysis ----------------
async def enhanced_analyze_openai(market):
    if not client:
        return None
    analysis_parts = []
    for symbol, data in market.items():
        if not data.get("price"): continue
        part = f"\n{symbol}: Price={data['price']}, RSI={data.get('rsi')}, MA_Short={data.get('ma_short')}, MA_Long={data.get('ma_long')}, VolSpike={data.get('volume_spike')}"
        analysis_parts.append(part)
    prompt = f"""
You are an expert crypto trading analyst. 
For each strong signal:
- ACTION (BUY/SELL)
- ENTRY price (near current or breakout)
- STOPLOSS (recent swing or S/R)
- TARGET (≥1:2 risk-reward)
- REASON with technical justification
- CONFIDENCE (70–100%)

Output format (strict, one line per strong signal):
SYMBOL - ACTION - ENTRY: <price> - SL: <price> - TP: <price> - REASON: <reason> - CONF: XX%

Market Data:
{"".join(analysis_parts)}
"""
    try:
        loop = asyncio.get_running_loop()
        def call_model():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1
            )
        resp = await loop.run_in_executor(None, call_model)
        choice = resp.choices[0]
        return choice.message.content if hasattr(choice, "message") else choice.text
    except Exception as e:
        print("OpenAI error", e)
        return None


# ---------------- Parsing ----------------
def enhanced_parse(text):
    out = {}
    if not text: return out
    for line in text.splitlines():
        line = line.strip()
        if not line or not any(x in line.upper() for x in ["BUY", "SELL"]):
            continue
        parts = [p.strip() for p in line.split(" - ")]
        if len(parts) < 6: continue
        symbol = parts[0].upper()
        action = parts[1].upper()
        entry = sl = tp = None; reason = ""; conf = None
        for p in parts[2:]:
            if p.upper().startswith("ENTRY:"):
                try: entry = float(p.split(":")[1].strip())
                except: pass
            elif p.upper().startswith("SL:"):
                try: sl = float(p.split(":")[1].strip())
                except: pass
            elif p.upper().startswith("TP:"):
                try: tp = float(p.split(":")[1].strip())
                except: pass
            elif p.upper().startswith("REASON:"):
                reason = p.split(":",1)[1].strip()
            elif "CONF" in p.upper():
                digits = "".join(c for c in p if c.isdigit())
                if digits: conf = int(digits)
        if action in ["BUY","SELL"] and conf and conf >= 70:
            out[symbol] = {
                "action": action, "entry": entry, "sl": sl, "tp": tp,
                "reason": reason, "confidence": conf, "timestamp": datetime.now()
            }
    return out


# ---------------- Telegram ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    await session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(path, "rb") as f:
        data = aiohttp.FormData()
        data.add_field("chat_id", str(TELEGRAM_CHAT_ID))
        data.add_field("caption", caption)
        data.add_field("photo", f, filename=os.path.basename(path), content_type="image/png")
        await session.post(url, data=data)


# ---------------- Main Loop ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession() as session:
        iteration = 0
        while True:
            iteration += 1
            print(f"\nIteration {iteration} {datetime.now()}")
            results = await asyncio.gather(*[fetch_enhanced_data(session, s) for s in SYMBOLS])
            market = {s:r for s,r in zip(SYMBOLS,results) if r and r.get("price")}
            analysis_result = await enhanced_analyze_openai(market)
            signals = enhanced_parse(analysis_result)
            for symbol, signal in signals.items():
                entry, sl, tp = signal["entry"], signal["sl"], signal["tp"]
                caption = f"""🚨 *{symbol}* → *{signal['action']}*
💰 Entry: {entry}
🛑 SL: {sl}
🎯 TP: {tp}
📊 Conf: {signal['confidence']}%
💡 {signal['reason']}"""
                await send_text(session, caption)
            await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    print("🚀 Starting GPT-driven Crypto Bot...")
    try:
        asyncio.run(enhanced_loop())
    except KeyboardInterrupt:
        print("Stopped by user")
