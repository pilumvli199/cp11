# main.py - Phase 4.6
# Multiframe analysis (30m,1h,4h) + strong-signal-only alerts + charts + Entry/SL/TP + cooldowns
# Fix: use candle timestamps (times_30m etc.) so charts render correctly on x-axis.

import os, asyncio, aiohttp, time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile
import math

load_dotenv()

# ---------- CONFIG ----------
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 1800))   # 30m default
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", 3600))     # 1 hour cooldown for same strong alert
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")             # optional
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=50"
OI_URL = "https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
LONGSHORT_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=30m&limit=1"

# ---------- Helpers: Telegram (debug) ----------
async def send_text(session, text):
    print(f"[DEBUG] send_text -> {text[:120]}")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Missing TELEGRAM_BOT_TOKEN/CHAT_ID")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"Markdown"}, timeout=15) as r:
            txt = await r.text()
            print(f"[DEBUG] sendMessage status={r.status} resp={txt[:800]}")
    except Exception as e:
        print("[ERROR] send_text exception:", repr(e))

async def send_photo(session, caption, path):
    print(f"[DEBUG] send_photo called: {path}")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Missing TELEGRAM_BOT_TOKEN/CHAT_ID")
        return False
    if not path or not os.path.exists(path):
        print("[ERROR] image missing:", path)
        return False
    try:
        size = os.path.getsize(path)
        print(f"[DEBUG] image size={size} bytes")
    except Exception as e:
        print("[WARN] stat failed:", e)
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
                return resp.status == 200
    except Exception as e:
        print("[ERROR] send_photo exception:", repr(e))
        return False
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
                print("[DEBUG] removed temp image", path)
        except Exception as e:
            print("[WARN] failed to remove temp image:", e)

# ---------- Fetch ----------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=20) as r:
            if r.status != 200:
                txt = await r.text()
                print(f"[WARN] HTTP {r.status} for {url} -> {txt[:200]}")
                return None
            return await r.json()
    except Exception as e:
        print("[WARN] fetch_json exception:", e, url)
        return None

async def fetch_symbol(session, symbol):
    # fetch 30m,1h,4h candles + ticker + oi + long/short
    t = await fetch_json(session, TICKER_URL.format(symbol=symbol))
    raw30 = await fetch_json(session, CANDLE_URL.format(symbol=symbol, interval="30m"))
    raw1  = await fetch_json(session, CANDLE_URL.format(symbol=symbol, interval="1h"))
    raw4  = await fetch_json(session, CANDLE_URL.format(symbol=symbol, interval="4h"))
    oi = await fetch_json(session, OI_URL.format(symbol=symbol))
    ls = await fetch_json(session, LONGSHORT_URL.format(symbol=symbol))
    out = {}
    if t:
        try:
            out["price"] = float(t.get("lastPrice", 0))
            out["volume"] = float(t.get("volume", 0))
        except:
            out["price"] = None
            out["volume"] = None

    def to_candles_and_times(raw):
        if not isinstance(raw, list):
            return [], []
        candles = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in raw]
        times = [int(x[0]) // 1000 for x in raw]  # openTime (ms -> s)
        return candles, times

    c30, t30 = to_candles_and_times(raw30)
    c1,  t1  = to_candles_and_times(raw1)
    c4,  t4  = to_candles_and_times(raw4)

    out["candles_30m"] = c30; out["times_30m"] = t30
    out["candles_1h"]  = c1;  out["times_1h"]  = t1
    out["candles_4h"]  = c4;  out["times_4h"]  = t4

    out["oi"] = float(oi.get("openInterest")) if isinstance(oi, dict) and oi.get("openInterest") else None
    if isinstance(ls, list) and ls:
        try:
            out["long_short_ratio"] = float(ls[0].get("longShortRatio", 1))
        except:
            out["long_short_ratio"] = None
    return out

# ---------- Local simple multiframe bias ----------
def simple_tf_bias(candles, lookback=10, threshold_pct=0.002):
    # returns "BUY"/"SELL"/"NEUTRAL"
    if not candles or len(candles) < 3:
        return "NEUTRAL"
    arr = candles[-lookback:] if len(candles) >= lookback else candles[:]
    closes = [c[3] for c in arr]
    last = closes[-1]
    mean = sum(closes)/len(closes)
    if mean == 0: return "NEUTRAL"
    pct = (last - mean) / mean
    if pct > threshold_pct:
        return "BUY"
    if pct < -threshold_pct:
        return "SELL"
    return "NEUTRAL"

# ---------- level plot & trade levels ----------
def calc_levels_impl(candles, lookback=24):
    if not candles: return (None,None,None)
    arr = candles[-lookback:] if len(candles)>=lookback else candles[:]
    highs = sorted([c[1] for c in arr], reverse=True)
    lows  = sorted([c[2] for c in arr])
    k = min(3, len(arr))
    res = sum(highs[:k]) / k
    sup = sum(lows[:k]) / k
    mid = (res + sup) / 2
    return sup, res, mid

def plot_candles(times, candles, levels, symbol):
    try:
        # times: list of epoch seconds or None
        if times and len(times) == len(candles):
            dates = [datetime.utcfromtimestamp(t) for t in times]
        else:
            dates = [datetime.utcfromtimestamp(int(time.time()) - (len(candles)-i)*60*30) for i in range(len(candles))]  # fallback spaced 30m
        o = [c[0] for c in candles]; h = [c[1] for c in candles]; l = [c[2] for c in candles]; cclose = [c[3] for c in candles]
        x = date2num(dates)
        fig, ax = plt.subplots(figsize=(8,4), dpi=100)
        for xi, oi, hi, li, ci in zip(x, o, h, l, cclose):
            col = "green" if ci >= oi else "red"
            ax.vlines(xi, li, hi, color="black", linewidth=0.6)
            ax.add_patch(plt.Rectangle((xi-0.3, min(oi,ci)), 0.6, abs(ci-oi), color=col, alpha=0.9))
        sup, res, mid = levels
        if res: ax.axhline(res, color="orange", linestyle="--", linewidth=1)
        if sup: ax.axhline(sup, color="purple", linestyle="--", linewidth=1)
        if mid: ax.axhline(mid, color="gray", linestyle=":")
        ax.set_title(symbol)
        fig.autofmt_xdate()
        tmp = NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, bbox_inches="tight")
        plt.close(fig)
        print("[DEBUG] chart saved to", tmp.name)
        return tmp.name
    except Exception as e:
        print("[ERROR] plot_candles exception:", e)
        return None

def compute_trade_levels(price, levels, bias):
    sup, res, mid = levels
    if not price: return None
    buf = 0.003
    if bias == "BUY":
        sl = (sup * (1 - buf)) if sup else price * 0.99
        risk = price - sl if price > sl else max(price*0.01, 1.0)
        tp1 = price + risk
        tp2 = price + 2*risk
        return {"entry": price, "sl": sl, "tp1": tp1, "tp2": tp2}
    if bias == "SELL":
        sl = (res * (1 + buf)) if res else price * 1.01
        risk = sl - price if sl > price else max(price*0.01, 1.0)
        tp1 = price - risk
        tp2 = price - 2*risk
        return {"entry": price, "sl": sl, "tp1": tp1, "tp2": tp2}
    return None

# ---------- OpenAI structured analysis (optional) ----------
async def openai_analysis(market_map):
    if not client:
        return None
    parts = []
    for s in SYMBOLS:
        d = market_map.get(s, {})
        parts.append(f"{s}: price={d.get('price','NA')} vol={d.get('volume','NA')} oi={d.get('oi','NA')} ls={d.get('long_short_ratio','NA')}")
        for tf in ("candles_30m","candles_1h","candles_4h"):
            if d.get(tf):
                last10 = d[tf][-10:]
                ct = ",".join([f"[{c[0]},{c[1]},{c[2]},{c[3]}]" for c in last10])
                parts.append(f"{s} {tf} last10: {ct}")
    prompt = (
        "You are a concise crypto technical analyst. For each symbol output one line:\n"
        "SYMBOL - BIAS - TIMEFRAMES - REASON\n"
        "Example:\nBTCUSDT - BUY - 30m,1h - ascending triangle breakout with rising volume\n\nNow analyze:\n"
        + "\n".join(parts)
    )
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=800, temperature=0.15
            )
        )
        text = resp.choices[0].message.content.strip()
        print("[DEBUG] OpenAI returned (truncated):", text[:800])
        parsed = {}
        for ln in text.splitlines():
            if " - " in ln:
                p = [x.strip() for x in ln.split(" - ")]
                if len(p) >= 3:
                    sym = p[0].upper()
                    bias_raw = p[1].upper()
                    tfs = p[2]
                    reason = p[3] if len(p) > 3 else ""
                    if "BULL" in bias_raw or "BUY" in bias_raw or "LONG" in bias_raw:
                        bias = "BUY"
                    elif "BEAR" in bias_raw or "SELL" in bias_raw or "SHORT" in bias_raw:
                        bias = "SELL"
                    else:
                        bias = "NEUTRAL"
                    parsed[sym] = {"bias": bias, "tfs": tfs, "reason": reason}
        return parsed
    except Exception as e:
        print("[WARN] openai_analysis failed:", e)
        return None

# ---------- strong-signal decision ----------
def decide_strong(local_biases, openai_bias, hist_biases):
    # history 3-cycle
    if len(hist_biases) >= 3 and hist_biases[-3:] == [hist_biases[-1]]*3 and hist_biases[-1] in ("BUY","SELL"):
        return True, hist_biases[-1], "3-cycle confirmation"
    # openai + local majority
    if openai_bias in ("BUY","SELL"):
        agrees = sum(1 for v in local_biases.values() if v == openai_bias)
        if agrees >= 2:
            return True, openai_bias, f"openai+local majority ({agrees}/3)"
    return False, None, ""

# ---------- main loop ----------
async def main_loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session, "*Bot online — Phase 4.6 (multiframe + strong alerts only)*")
        history = {s: [] for s in SYMBOLS}       # store last majority biases
        cooldowns = {s: 0 for s in SYMBOLS}      # timestamp until which don't resend same strong alert

        while True:
            start = time.time()
            tasks = [fetch_symbol(session, s) for s in SYMBOLS]
            results = await asyncio.gather(*tasks)
            market = {s:r for s,r in zip(SYMBOLS, results)}
            print("[DEBUG] fetched symbols:", list(market.keys()))

            # local multiframe biases
            local = {}
            for s,d in market.items():
                local[s] = {
                    "30m": simple_tf_bias(d.get("candles_30m", [])),
                    "1h": simple_tf_bias(d.get("candles_1h", [])),
                    "4h": simple_tf_bias(d.get("candles_4h", []))
                }
            print("[DEBUG] local biases sample:", {k: local[k] for k in ['BTCUSDT','ETHUSDT','SOLUSDT'] if k in local})

            # openai analysis (optional)
            openai_out = await openai_analysis(market) if client else None

            # send summary report (short) every cycle
            summary_lines = [f"Snapshot (UTC {datetime.utcnow().strftime('%H:%M')})"]
            for s in SYMBOLS:
                d = market.get(s, {})
                summary_lines.append(f"{s}: {d.get('price','NA')} vol={d.get('volume','NA')}")
            await send_text(session, "🧠 " + "\n".join(summary_lines))

            # per-symbol decision
            for s in SYMBOLS:
                d = market.get(s, {})
                openai_bias = None
                openai_reason = ""
                if openai_out and s in openai_out:
                    openai_bias = openai_out[s]["bias"]
                    openai_reason = openai_out[s]["reason"]
                loc = local.get(s, {})
                # majority of local tf
                vals = list(loc.values())
                majority = max(set(vals), key=vals.count) if vals else "NEUTRAL"
                history[s].append(majority)
                if len(history[s]) > 20: history[s].pop(0)

                strong, strong_bias, strong_note = decide_strong(loc, openai_bias, history[s])
                now_ts = time.time()
                if strong and strong_bias:
                    if now_ts < cooldowns[s]:
                        print(f"[DEBUG] cooldown active for {s}, skipping strong alert.")
                    else:
                        levels = calc_levels_impl(d.get("candles_30m", []))
                        tl = compute_trade_levels(d.get("price"), levels, strong_bias)
                        caption = f"🚨 STRONG {strong_bias}: {s}\nTFs(local): 30m={loc.get('30m')},1h={loc.get('1h')},4h={loc.get('4h')}\nReason(openai): {openai_reason or 'N/A'}\nConfirm: {strong_note}"
                        if tl:
                            caption += f"\nEntry:{tl['entry']:.4f} SL:{tl['sl']:.4f} TP1:{tl['tp1']:.4f} TP2:{tl['tp2']:.4f}"
                        c30 = d.get("candles_30m", [])
                        t30 = d.get("times_30m", None)
                        chart_path = plot_candles(t30, c30, levels, s) if c30 else None
                        if chart_path:
                            ok = await send_photo(session, caption, chart_path)
                            print(f"[DEBUG] sent strong alert for {s}, ok={ok}")
                        else:
                            await send_text(session, caption)
                        cooldowns[s] = now_ts + COOLDOWN_SEC
                else:
                    # show informative chart for top symbols
                    if s in ("BTCUSDT","ETHUSDT","SOLUSDT"):
                        levels = calc_levels_impl(d.get("candles_30m", []))
                        tl = compute_trade_levels(d.get("price"), levels, majority if majority in ("BUY","SELL") else None)
                        caption = f"ℹ️ {s} {majority} (local majority) · OpenAI:{openai_bias or 'N/A'}\nTFs:30m={loc.get('30m')},1h={loc.get('1h')},4h={loc.get('4h')}"
                        if tl:
                            caption += f"\nEntry:{tl['entry']:.4f} SL:{tl['sl']:.4f} TP1:{tl['tp1']:.4f}"
                        c30 = d.get("candles_30m", [])
                        t30 = d.get("times_30m", None)
                        try:
                            chart_path = plot_candles(t30, c30, levels, s) if c30 else None
                            if chart_path:
                                await send_photo(session, caption, chart_path)
                            else:
                                await send_text(session, caption)
                        except Exception as e:
                            print("[WARN] info chart send failed:", e)
                            await send_text(session, caption)

            # cycle end
            elapsed = time.time() - start
            to_sleep = max(0, POLL_INTERVAL - elapsed)
            print(f"[DEBUG] cycle finished. sleeping {to_sleep} sec.")
            await asyncio.sleep(to_sleep)

# run
if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("Interrupted")
