# main.py - Phase 4.4 Debug-enabled (Charts + Entry/SL/TP) with bias normalization
import os, asyncio, aiohttp, time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile

load_dotenv()

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT"]
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 1800))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=50"

# ---- Telegram helpers with debug ----
async def send_text(session, text):
    print(f"[DEBUG] send_text -> {text[:120]}")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"Markdown"}, timeout=15) as r:
            txt = await r.text()
            print(f"[DEBUG] sendMessage status={r.status} resp={txt[:800]}")
    except Exception as e:
        print("[ERROR] send_text exception:", repr(e))

async def send_photo(session, caption, path):
    print(f"[DEBUG] send_photo called. path={path} caption={caption[:120]}")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing")
        return False
    if not path or not os.path.exists(path):
        print("[ERROR] Image path missing or not exists:", path)
        return False
    try:
        size = os.path.getsize(path)
        print(f"[DEBUG] image exists size={size} bytes")
    except Exception as e:
        print("[ERROR] could not stat image:", e)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("chat_id", str(TELEGRAM_CHAT_ID))
            data.add_field("caption", caption)
            data.add_field("parse_mode", "Markdown")
            data.add_field("photo", f, filename=os.path.basename(path), content_type="image/png")
            async with session.post(url, data=data, timeout=30) as resp:
                txt = await resp.text()
                print(f"[DEBUG] sendPhoto status={resp.status} response={txt[:1500]}")
                if resp.status != 200:
                    print("[ERROR] sendPhoto failed ->", resp.status, txt)
                    return False
                return True
    except Exception as e:
        print("[ERROR] send_photo exception:", repr(e))
        return False
    finally:
        # try remove the file (caller may also remove) but don't raise
        try:
            if os.path.exists(path):
                os.remove(path)
                print("[DEBUG] removed temp image:", path)
        except Exception as e:
            print("[WARN] failed to remove temp image:", e)

# ---- Binance fetch helpers ----
async def fetch_json(session,url):
    try:
        async with session.get(url, timeout=15) as r:
            if r.status != 200:
                txt = await r.text()
                print(f"[WARN] HTTP {r.status} for {url} -> {txt[:200]}")
                return None
            return await r.json()
    except Exception as e:
        print("[WARN] fetch_json exception for", url, ":", repr(e))
        return None

async def fetch_data(session,symbol):
    t = await fetch_json(session, TICKER_URL.format(symbol=symbol))
    c30 = await fetch_json(session, CANDLE_URL.format(symbol=symbol, interval="30m"))
    out = {}
    if t:
        try:
            out["price"] = float(t.get("lastPrice", 0))
            out["volume"] = float(t.get("volume", 0))
        except:
            out["price"] = None
            out["volume"] = None
    if isinstance(c30, list):
        out["candles"] = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in c30]
        out["times"] = [int(x[0])//1000 for x in c30]
    return out

# ---- levels/plot/trade helpers ----
def levels(candles,lookback=24):
    if not candles: return (None,None,None)
    arr = candles[-lookback:] if len(candles)>=lookback else candles[:]
    highs = sorted([c[1] for c in arr], reverse=True)
    lows = sorted([c[2] for c in arr])
    k = min(3, len(arr))
    res = sum(highs[:k]) / k
    sup = sum(lows[:k]) / k
    mid = (res + sup) / 2
    return sup, res, mid

def plot_chart(times, candles, sym, levs):
    try:
        dates = [datetime.utcfromtimestamp(t) for t in times]
        o = [c[0] for c in candles]; h = [c[1] for c in candles]; l = [c[2] for c in candles]; cvals = [c[3] for c in candles]
        x = date2num(dates)
        fig, ax = plt.subplots(figsize=(7,4), dpi=100)
        for xi, oi, hi, li, ci in zip(x, o, h, l, cvals):
            col = "green" if ci >= oi else "red"
            ax.vlines(xi, li, hi, color="black")
            ax.add_patch(plt.Rectangle((xi-0.2, min(oi,ci)), 0.4, abs(ci-oi), color=col))
        sup, res, mid = levs
        if res: ax.axhline(res, color="orange", linestyle="--")
        if sup: ax.axhline(sup, color="purple", linestyle="--")
        if mid: ax.axhline(mid, color="gray", linestyle=":")
        ax.set_title(sym)
        fig.autofmt_xdate()
        tmp = NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, bbox_inches="tight")
        plt.close(fig)
        print("[DEBUG] chart saved to", tmp.name)
        return tmp.name
    except Exception as e:
        print("[ERROR] plot_chart exception:", repr(e))
        return None

def trade_levels(price, levs, bias):
    sup, res, mid = levs
    if not price: return None
    if bias == "BUY":
        sl = sup * 0.997 if sup else price * 0.99
        risk = price - sl
        return {"entry": price, "sl": sl, "tp1": price + risk, "tp2": price + 2 * risk}
    if bias == "SELL":
        sl = res * 1.003 if res else price * 1.01
        risk = sl - price
        return {"entry": price, "sl": sl, "tp1": price - risk, "tp2": price - 2 * risk}
    return None

# ---- OpenAI analysis ----
async def analyze_openai(market):
    if not client:
        print("[WARN] OpenAI client not configured")
        return None
    try:
        parts = []
        for s, d in market.items():
            parts.append(f"{s}: price={d.get('price')} vol={d.get('volume')}")
            if d.get("candles"):
                last10 = d["candles"][-10:]
                parts.append(f"{s} 30m last10:" + ",".join([f"[{c[0]},{c[1]},{c[2]},{c[3]}]" for c in last10]))
        # Ask for BUY/SELL/NEUTRAL but GPT may still use synonyms; we'll normalize later.
        prompt = ("Output lines: SYMBOL - BIAS - TF - REASON\\n" + "\\n".join(parts))
        print("[DEBUG] OpenAI prompt length:", len(prompt))
        resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=500, temperature=0.2
        ))
        text = resp.choices[0].message.content.strip()
        print("[DEBUG] OpenAI returned (first 500 chars):", text[:500])
        return text
    except Exception as e:
        print("[ERROR] analyze_openai exception:", repr(e))
        return None

# ---- normalization utility ----
def normalize_bias(raw_bias: str) -> str:
    if not raw_bias:
        return "NEUTRAL"
    b = raw_bias.strip().upper()
    # map common synonyms to BUY/SELL/NEUTRAL
    if b in ("BUY", "LONG", "BULLISH", "BULL", "ACCUMULATE", "GO LONG"):
        return "BUY"
    if b in ("SELL", "SHORT", "BEARISH", "BEAR", "LIQUIDATE", "GO SHORT"):
        return "SELL"
    if b in ("NEUTRAL", "HOLD", "WAIT", "NO ACTION"):
        return "NEUTRAL"
    # sometimes GPT returns phrases like "slightly bullish" -> check keywords
    if "BULL" in b or "BUY" in b:
        return "BUY"
    if "BEAR" in b or "SELL" in b or "SHORT" in b:
        return "SELL"
    return "NEUTRAL"

def parse(text):
    out = {}
    if not text: return out
    for ln in text.splitlines():
        parts = [p.strip() for p in ln.split(" - ")]
        if len(parts) >= 3:
            sym = parts[0].upper()
            raw_bias = parts[1]
            bias = normalize_bias(raw_bias)
            tfs = parts[2]
            reason = parts[3] if len(parts) > 3 else ""
            out[sym] = {"bias": bias, "raw_bias": raw_bias, "tf": tfs, "reason": reason}
    return out

# ---- main loop ----
async def loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session, "Bot online Phase-4.4 (debug)")
        prev = {}
        while True:
            try:
                tasks = [fetch_data(session, s) for s in SYMBOLS]
                res = await asyncio.gather(*tasks)
                market = {s:r for s,r in zip(SYMBOLS,res)}
                print("[DEBUG] fetched market keys:", list(market.keys()))
                analysis = await analyze_openai(market)
                parsed = parse(analysis) if analysis else {}
                print("[DEBUG] parsed keys:", parsed.keys())
                for s, info in parsed.items():
                    bias = info.get("bias","NEUTRAL")
                    raw_bias = info.get("raw_bias","")
                    d = market.get(s, {})
                    levs = levels(d.get("candles", []))
                    # send chart+trade for BUY/SELL (including normalized from BULLISH/BEARISH)
                    if bias in ("BUY","SELL"):
                        tl = trade_levels(d.get("price"), levs, bias)
                        caption = f"🚨 {s} {bias} (raw:{raw_bias}) TF:{info.get('tf')}\\nReason:{info.get('reason')}\\n"
                        if tl:
                            caption += f"Entry:{tl['entry']:.2f} SL:{tl['sl']:.2f} TP1:{tl['tp1']:.2f} TP2:{tl['tp2']:.2f}"
                        chart_path = None
                        try:
                            chart_path = plot_chart(d.get("times", []), d.get("candles", []), s, levs)
                            if chart_path:
                                ok = await send_photo(session, caption, chart_path)
                                print(f"[DEBUG] send_photo ok={ok} for {s}")
                            else:
                                print("[WARN] no chart_path generated for", s)
                                await send_text(session, caption)
                        except Exception as e:
                            print("[ERROR] during chart/send for", s, repr(e))
                            await send_text(session, caption)
                    else:
                        # NEUTRAL -> send short analysis text only
                        await send_text(session, f"ℹ️ {s} NEUTRAL · reason: {info.get('reason')} (raw:{raw_bias})")
                print("[DEBUG] cycle done, sleeping", POLL_INTERVAL)
            except Exception as e:
                print("[ERROR] main loop exception:", repr(e))
            await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    asyncio.run(loop())
