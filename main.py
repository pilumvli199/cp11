# main_improved.py - Phase 5 (11 coins) - improved robustness & fixes
import os
import asyncio
import aiohttp
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile
import numpy as np
import math
import shutil

load_dotenv()

# --- Config ---
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]
POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", 1800)))  # at least 30s
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 65.0))

# instantiate OpenAI client if key present
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=50"

# ---------------- Telegram helpers ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured. Skipping send_text.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}) as r:
            if r.status != 200:
                txt = await r.text()
                print(f"Telegram send_text failed {r.status}: {txt}")
    except Exception as e:
        print("send_text error:", e)

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured. Skipping send_photo.")
        # cleanup file anyway
        try:
            os.remove(path)
        except:
            pass
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("chat_id", str(TELEGRAM_CHAT_ID))
            data.add_field("caption", caption)
            data.add_field("photo", f, filename=os.path.basename(path), content_type="image/png")
            async with session.post(url, data=data, timeout=60) as r:
                if r.status != 200:
                    text = await r.text()
                    print(f"Telegram send_photo failed {r.status}: {text}")
    except Exception as e:
        print("send_photo error:", e)
    finally:
        # ensure temp file removed
        try:
            os.remove(path)
        except Exception:
            pass

# ---------------- Fetching ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=15) as r:
            if r.status != 200:
                # print debug info then return None
                try:
                    txt = await r.text()
                except:
                    txt = "<no body>"
                print(f"fetch_json {url} returned {r.status}: {txt[:200]}")
                return None
            return await r.json()
    except Exception as e:
        print("fetch_json exception for", url, e)
        return None

async def fetch_data(session, symbol):
    t = await fetch_json(session, TICKER_URL.format(symbol=symbol))
    c = await fetch_json(session, CANDLE_URL.format(symbol=symbol))
    out = {}
    if t:
        try:
            out["price"] = float(t.get("lastPrice", 0))
            out["volume"] = float(t.get("volume", 0))
        except Exception:
            out["price"] = None
            out["volume"] = None
    if isinstance(c, list):
        # Binance kline: [openTime, open, high, low, close, ...]
        try:
            out["candles"] = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in c]
            out["times"] = [int(x[0]) // 1000 for x in c]
        except Exception as e:
            print("candle parse error for", symbol, e)
            out["candles"] = None
            out["times"] = None
    return out

# ---------------- Levels / Trendlines ----------------
def levels(candles, lookback=24):
    if not candles or len(candles) < 3:
        return (None, None, None)
    arr = candles[-lookback:] if len(candles) >= lookback else candles
    highs = sorted([c[1] for c in arr], reverse=True)
    lows = sorted([c[2] for c in arr])
    k = min(3, len(arr))
    res = sum(highs[:k]) / k if highs else None
    sup = sum(lows[:k]) / k if lows else None
    mid = (res + sup) / 2 if (res is not None and sup is not None) else None
    return sup, res, mid

def trendline(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return None
    try:
        coeffs = np.polyfit(xs, ys, 1)
        return coeffs  # m, b
    except Exception as e:
        print("trendline polyfit error:", e)
        return None

def plot_chart(times, candles, sym, levs):
    if not times or not candles or len(times) != len(candles) or len(candles) < 2:
        raise ValueError("Insufficient data to plot")
    dates = [datetime.utcfromtimestamp(int(t)) for t in times]
    o = [c[0] for c in candles]; h = [c[1] for c in candles]; l = [c[2] for c in candles]; c_ = [c[3] for c in candles]
    x = date2num(dates)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)

    # Black & White candlesticks
    width = 0.6 * (x[1] - x[0]) if len(x) > 1 else 0.4
    for xi, oi, hi, li, ci in zip(x, o, h, l, c_):
        col = "white" if ci >= oi else "black"
        edge = "black"
        ax.vlines(xi, li, hi, color="black", linewidth=0.6)
        low = min(oi, ci)
        rect = plt.Rectangle((xi - width / 2, low), width, abs(ci - oi) if abs(ci - oi) > 0.0000001 else 0.0001,
                             facecolor=col, edgecolor=edge)
        ax.add_patch(rect)

    sup, res, mid = levs or (None, None, None)
    if res is not None:
        ax.axhline(res, color="orange", linestyle="--", label=f"Res {res:.6f}" if abs(res) < 1 else f"Res {res:.2f}")
    if sup is not None:
        ax.axhline(sup, color="purple", linestyle="--", label=f"Sup {sup:.6f}" if abs(sup) < 1 else f"Sup {sup:.2f}")
    if mid is not None:
        ax.axhline(mid, color="gray", linestyle=":", label=f"Mid {mid:.2f}")

    # Trendline approx (last up to 10 closes)
    n = min(10, len(c_))
    if n >= 2:
        ys = np.array(c_[-n:])
        xs_tr = np.arange(n)
        coeffs = trendline(xs_tr, ys)
        if coeffs is not None:
            m, b = coeffs
            yy = m * xs_tr + b
            # align to date x positions for the last n points
            ax.plot(x[-n:], yy, linestyle="-.", label="Trendline")

    ax.set_title(sym)
    ax.legend(loc="upper left", fontsize="small")
    fig.autofmt_xdate()
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

# ---------------- OpenAI analysis ----------------
async def analyze_openai(market):
    """
    market: dict of symbol -> {price, volume, candles, times}
    Returns raw text from model or None.
    """
    if not client:
        print("No OpenAI client configured.")
        return None

    # Build data parts for symbols that have at least price info.
    parts = []
    for s, d in market.items():
        price = d.get("price")
        vol = d.get("volume")
        if price is None:
            continue
        parts.append(f"{s}: price={price} vol={vol}")
        if d.get("candles"):
            last10 = d["candles"][-10:]
            row = ",".join([f"[{c[0]},{c[1]},{c[2]},{c[3]}]" for c in last10])
            parts.append(f"{s} 30m last10: {row}")

    if not parts:
        print("No market data available to send to OpenAI.")
        return None

    prompt = (
        "You are an advanced crypto analyst.\n"
        "For each symbol analyze candlesticks, chart patterns, support/resistance, trendlines, volume, and detect BUY/SELL opportunities only if strong signals appear.\n"
        "Output format per line EXACTLY (one per symbol you comment on):\n"
        "SYMBOL - BIAS - REASON - CONF: <NN>%\n"
        "Where BIAS = BUY/SELL/NEUTRAL, CONF = 0-100 confidence.\n"
        "Reason must briefly mention detected patterns/levels.\n\n"
        "Data:\n" + "\n".join(parts)
    )

    # call OpenAI in executor to avoid blocking event-loop (client may be blocking)
    try:
        loop = asyncio.get_running_loop()
        def call_model():
            # Use chat completion style that the 'OpenAI' client supports.
            # This may be provider-specific; adapt if your SDK differs.
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=900,
                temperature=0.2
            )
        resp = await loop.run_in_executor(None, call_model)
        # Safely extract text
        try:
            # Adapt to response shape: resp.choices[0].message.content or resp.choices[0].text
            choice = resp.choices[0]
            if hasattr(choice, "message"):
                content = choice.message.content
            else:
                # fallback
                content = getattr(choice, "text", None)
            if content is None:
                content = str(resp)
            return content.strip()
        except Exception:
            return str(resp)
    except Exception as e:
        print("OpenAI call failed:", e)
        traceback.print_exc()
        return None

def parse(text):
    """
    Parse model output into dict: {SYMBOL: {"bias":..., "reason":..., "conf": int or None}}
    Accepts lines like:
      BTCUSDT - BUY - breakout above res - CONF: 78%
    """
    out = {}
    if not text:
        return out
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # Try split into 3 parts at " - "
        parts = [p.strip() for p in ln.split(" - ", 3)]
        if len(parts) < 3:
            # try splitting by " - " less strictly
            parts = [p.strip() for p in ln.split(" - ")]
        if len(parts) < 3:
            continue
        sym = parts[0].upper()
        bias = parts[1].upper()
        reason = parts[2]
        conf = None
        # find CONF ... pattern anywhere in the line
        up = ln.upper()
        if "CONF" in up:
            try:
                # find segment like CONF[: ] 72%
                idx = up.index("CONF")
                tail = up[idx:]
                # extract digits
                digits = ""
                for ch in tail:
                    if ch.isdigit():
                        digits += ch
                    elif digits:
                        break
                if digits:
                    conf = int(digits)
            except Exception:
                conf = None
        out[sym] = {"bias": bias, "reason": reason, "conf": conf}
    return out

# ---------------- Loop ----------------
async def loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session, f"Bot online — Phase-5 (symbols={len(SYMBOLS)}, conf≥{SIGNAL_CONF_THRESHOLD}%, poll={POLL_INTERVAL}s)")
        while True:
            try:
                # Fetch concurrently
                tasks = [fetch_data(session, s) for s in SYMBOLS]
                res = await asyncio.gather(*tasks)
                market = {s: r for s, r in zip(SYMBOLS, res)}
                # ask model
                txt = await analyze_openai(market)
                if not txt:
                    print("Model returned nothing; skipping this cycle.")
                else:
                    parsed = parse(txt)
                    # iterate parsed signals
                    for s, info in parsed.items():
                        try:
                            conf = info.get("conf") or 0
                            bias = (info.get("bias") or "NEUTRAL").upper()
                            if conf >= SIGNAL_CONF_THRESHOLD and bias in ("BUY", "SELL"):
                                d = market.get(s, {})
                                candles = d.get("candles")
                                times = d.get("times")
                                if not candles or not times or len(candles) < 5:
                                    print(f"Skipping plotting for {s}: insufficient data")
                                    continue
                                levs = levels(candles)
                                try:
                                    chart = plot_chart(times, candles, s, levs)
                                except Exception as e:
                                    print("plot_chart failed for", s, e)
                                    continue
                                caption = f"🚨 {s} → {bias}\nReason: {info.get('reason')}\nConf: {conf}%"
                                await send_photo(session, caption, chart)
                        except Exception as e:
                            print("Error processing parsed entry:", e)
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print("Main loop exception:", e)
                traceback.print_exc()
                # simple backoff on error
                await asyncio.sleep(min(60, POLL_INTERVAL))

if __name__ == "__main__":
    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        print("Exiting on user interrupt.")
