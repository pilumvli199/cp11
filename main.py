# main.py - Phase 4.4 (Charts + Entry/SL/TP) with signal filter (threshold + local+openai agreement)
import os, asyncio, aiohttp, time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile

load_dotenv()

# --- Config ---
SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT"]  # add more as needed
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 1800))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 65.0))

client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("[WARN] OpenAI init failed:", e)
        client = None
else:
    print("[WARN] OPENAI_API_KEY not set. OpenAI analysis disabled.")

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=50"

# ---------------- Telegram helpers ----------------
async def send_text(session, text):
    print(f"[TRACE] send_text -> {text[:120]}")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Telegram credentials missing")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"Markdown"}, timeout=15) as r:
            txt = await r.text()
            print(f"[TRACE] sendMessage status={r.status}")
            # don't print whole resp to avoid token leaks; useful for debugging
            if r.status != 200:
                print("[WARN] sendMessage response:", txt[:800])
    except Exception as e:
        print("[ERROR] send_text exception:", e)

async def send_photo(session, caption, path):
    print(f"[TRACE] send_photo path={path} caption_len={len(caption)}")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Telegram creds missing")
        return False
    if not path or not os.path.exists(path):
        print("[ERROR] Image not found:", path)
        return False
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
                print(f"[TRACE] sendPhoto status={resp.status}")
                if resp.status != 200:
                    print("[WARN] sendPhoto resp:", txt[:800])
                    return False
                return True
    except Exception as e:
        print("[ERROR] send_photo exception:", e)
        return False
    finally:
        try:
            if path and os.path.exists(path):
                os.remove(path)
                print("[TRACE] removed temp image:", path)
        except Exception as e:
            print("[WARN] failed to remove temp image:", e)

# ---------------- Fetching ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=15) as r:
            if r.status != 200:
                txt = await r.text()
                print(f"[WARN] HTTP {r.status} for {url} -> {txt[:200]}")
                return None
            return await r.json()
    except Exception as e:
        print("[WARN] fetch_json exception for", url, ":", e)
        return None

async def fetch_data(session, symbol):
    ticker = await fetch_json(session, TICKER_URL.format(symbol=symbol))
    c30 = await fetch_json(session, CANDLE_URL.format(symbol=symbol, interval="30m"))
    out = {}
    if ticker:
        try:
            out["price"] = float(ticker.get("lastPrice", 0))
            out["volume"] = float(ticker.get("volume", 0))
        except:
            out["price"] = None
            out["volume"] = None
    if isinstance(c30, list):
        out["candles"] = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in c30]
        out["times"] = [int(x[0])//1000 for x in c30]
    return out

# ---------------- Local simple bias detector ----------------
def local_bias_from_candles(candles):
    # Very lightweight heuristic:
    # If last 3 closes are strictly increasing => BUY
    # If last 3 closes are strictly decreasing => SELL
    # Else NEUTRAL
    if not candles or len(candles) < 3:
        return "NEUTRAL"
    closes = [c[3] for c in candles[-5:]]  # use last up to 5 closes
    # take last 3
    last3 = closes[-3:]
    if last3[0] < last3[1] < last3[2]:
        return "BUY"
    if last3[0] > last3[1] > last3[2]:
        return "SELL"
    return "NEUTRAL"

# ---------------- Levels / chart / trade helpers ----------------
def calc_levels(candles, lookback=24):
    if not candles:
        return (None, None, None)
    arr = candles[-lookback:] if len(candles) >= lookback else candles[:]
    highs = sorted([c[1] for c in arr], reverse=True)
    lows = sorted([c[2] for c in arr])
    k = min(3, max(1, len(arr)))
    res = sum(highs[:k]) / k
    sup = sum(lows[:k]) / k
    mid = (res + sup) / 2
    return (sup, res, mid)

def plot_chart(times, candles, symbol, levels):
    try:
        dates = [datetime.utcfromtimestamp(t) for t in times]
        o = [c[0] for c in candles]; h = [c[1] for c in candles]; l = [c[2] for c in candles]; cvals = [c[3] for c in candles]
        x = date2num(dates)
        fig, ax = plt.subplots(figsize=(8,4), dpi=100)
        for xi, oi, hi, li, ci in zip(x, o, h, l, cvals):
            col = "green" if ci >= oi else "red"
            ax.vlines(xi, li, hi, color="black", linewidth=0.6)
            ax.add_patch(plt.Rectangle((xi-0.2, min(oi, ci)), 0.4, abs(ci-oi), color=col, alpha=0.9))
        sup, res, mid = levels
        if res is not None: ax.axhline(res, color="orange", linestyle="--", linewidth=0.9, label=f"res {res:.2f}")
        if sup is not None: ax.axhline(sup, color="purple", linestyle="--", linewidth=0.9, label=f"sup {sup:.2f}")
        if mid is not None: ax.axhline(mid, color="gray", linestyle=":", linewidth=0.7, label=f"mid {mid:.2f}")
        ax.set_title(symbol)
        ax.legend(loc="upper left", fontsize="small")
        fig.autofmt_xdate()
        tmp = NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, bbox_inches="tight")
        plt.close(fig)
        print("[TRACE] chart saved to", tmp.name)
        return tmp.name
    except Exception as e:
        print("[ERROR] plot_chart failed:", e)
        return None

def compute_trade_levels(price, levels, bias):
    if not price:
        return None
    sup, res, mid = levels
    buf = 0.003
    if bias == "BUY":
        sl = sup * (1 - buf) if sup else price * 0.99
        risk = price - sl if price > sl else max(price*0.01, 1.0)
        tp1 = price + 1 * risk
        tp2 = price + 2 * risk
        return {"entry": price, "sl": sl, "tp1": tp1, "tp2": tp2}
    if bias == "SELL":
        sl = res * (1 + buf) if res else price * 1.01
        risk = sl - price if sl > price else max(price*0.01, 1.0)
        tp1 = price - 1 * risk
        tp2 = price - 2 * risk
        return {"entry": price, "sl": sl, "tp1": tp1, "tp2": tp2}
    return None

# ---------------- OpenAI analysis (structured with CONF) ----------------
async def analyze_openai(market_map: dict):
    if not client:
        print("[WARN] OpenAI not configured")
        return None
    try:
        parts = []
        for s in SYMBOLS:
            d = market_map.get(s) or {}
            parts.append(f"{s}: price={d.get('price','NA')} vol={d.get('volume','NA')}")
            if d.get("candles"):
                last10 = d["candles"][-10:]
                ct = ",".join([f"[{c[0]},{c[1]},{c[2]},{c[3]}]" for c in last10])
                parts.append(f"{s} 30m last10: {ct}")
        prompt = (
            "You are a concise crypto technical analyst. For each symbol provide exactly ONE LINE in the format:\n"
            "SYMBOL - BIAS - TF - REASON - CONF: <NN>%\n"
            "Where BIAS should be one of: BUY / SELL / NEUTRAL (but synonyms ok), TF is timeframe tokens like '30m' or '30m,1h'.\n"
            "CONF should be an integer percent representing your confidence in the bias (0-100).\n\nAnalyze the data below and output one line per symbol only.\n\n"
            + "\n".join(parts)
        )
        print("[TRACE] OpenAI prompt length:", len(prompt))
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=800, temperature=0.15,
            ),
        )
        text = resp.choices[0].message.content.strip()
        print("[TRACE] OpenAI returned (truncated):", text[:800])
        return text
    except Exception as e:
        print("[ERROR] analyze_openai failed:", e)
        return None

def parse_openai_structured(text: str):
    out = {}
    if not text:
        return out
    for ln in text.splitlines():
        if not ln.strip():
            continue
        # expected format: SYMBOL - BIAS - TF - REASON - CONF: NN%
        parts = [p.strip() for p in ln.split(" - ")]
        if len(parts) < 3:
            continue
        sym = parts[0].upper()
        raw_bias = parts[1]
        bias = normalize_bias(raw_bias)
        tfs = parts[2]
        reason = ""
        conf = None
        # reason may be in parts[3] and conf may be appended like " ... - CONF: 72%"
        if len(parts) >= 4:
            # join remaining parts except final if it contains CONF:
            tail = " - ".join(parts[3:])
            # try extract CONF: NN%
            import re
            m = re.search(r"CONF[: ]+([0-9]{1,3})%?", tail, re.IGNORECASE)
            if m:
                try:
                    conf = float(m.group(1))
                except:
                    conf = None
                # remove CONF segment from reason
                reason = re.sub(r"CONF[: ]+[0-9]{1,3}%?", "", tail, flags=re.IGNORECASE).strip(" -;")
            else:
                reason = tail
        out[sym] = {"bias": bias, "raw_bias": raw_bias, "tfs": tfs, "reason": reason, "conf": conf}
    return out

# normalize synonyms
def normalize_bias(raw_bias: str) -> str:
    if not raw_bias:
        return "NEUTRAL"
    b = raw_bias.strip().upper()
    if b in ("BUY", "LONG", "BULLISH", "BULL", "ACCUMULATE", "GO LONG"):
        return "BUY"
    if b in ("SELL", "SHORT", "BEARISH", "BEAR", "LIQUIDATE", "GO SHORT"):
        return "SELL"
    if b in ("NEUTRAL", "HOLD", "WAIT", "NO ACTION"):
        return "NEUTRAL"
    if "BULL" in b or "BUY" in b:
        return "BUY"
    if "BEAR" in b or "SELL" in b or "SHORT" in b:
        return "SELL"
    return "NEUTRAL"

# ---------------- Main loop ----------------
async def task_loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session, f"Bot online — Phase-4.4 (threshold {SIGNAL_CONF_THRESHOLD}%)")
        prev_market = {}
        while True:
            start = time.time()
            # fetch data
            tasks = [fetch_data(session, s) for s in SYMBOLS]
            res = await asyncio.gather(*tasks, return_exceptions=True)
            market_map = {}
            for s, r in zip(SYMBOLS, res):
                if isinstance(r, Exception) or r is None:
                    print(f"[WARN] fetch for {s} failed:", r)
                    market_map[s] = {}
                else:
                    market_map[s] = r
            # openai analysis
            openai_text = await analyze_openai(market_map) if client else None
            parsed = parse_openai_structured(openai_text) if openai_text else {}
            print("[TRACE] parsed openai keys:", list(parsed.keys()))
            # Evaluate each symbol: require both local and openai match and conf >= threshold
            for s in SYMBOLS:
                ai = parsed.get(s, {})
                ai_bias = ai.get("bias", "NEUTRAL")
                ai_conf = ai.get("conf", None)
                ai_reason = ai.get("reason", "")
                local = "NEUTRAL"
                candles = market_map.get(s, {}).get("candles")
                if candles:
                    local = local_bias_from_candles(candles)
                print(f"[TRACE] {s} local={local} openai={ai_bias} conf={ai_conf}")
                # only proceed if bias yes and conf numeric
                if ai_conf is None:
                    print(f"[DEBUG] skipping {s} because OpenAI conf missing")
                    continue
                try:
                    conf_val = float(ai_conf)
                except:
                    print(f"[DEBUG] skipping {s} because conf not numeric: {ai_conf}")
                    continue
                if conf_val < SIGNAL_CONF_THRESHOLD:
                    print(f"[DEBUG] skipping {s} conf {conf_val} < threshold {SIGNAL_CONF_THRESHOLD}")
                    continue
                # require agreement
                if local != ai_bias:
                    print(f"[DEBUG] skipping {s} bias mismatch local={local} ai={ai_bias}")
                    continue
                if ai_bias not in ("BUY", "SELL"):
                    print(f"[DEBUG] skipping {s} ai_bias not actionable: {ai_bias}")
                    continue
                # compose alert: compute levels, trade suggestions, chart and send
                price = market_map.get(s, {}).get("price")
                levels = calc_levels(candles or [])
                trade = compute_trade_levels(price, levels, ai_bias)
                caption_lines = [
                    f"🚨 STRONG SIGNAL: {s} → {ai_bias} (Conf: {int(conf_val)}%)",
                    f"Reason: {ai_reason or 'OpenAI signal'}",
                ]
                if trade:
                    caption_lines.append("")
                    caption_lines.append(f"Entry: {trade['entry']:.4f}")
                    caption_lines.append(f"SL: {trade['sl']:.4f}")
                    caption_lines.append(f"TP1: {trade['tp1']:.4f}")
                    caption_lines.append(f"TP2: {trade['tp2']:.4f}")
                caption = "\n".join(caption_lines)
                chart_path = None
                try:
                    chart_path = plot_chart(market_map.get(s, {}).get("times", []), candles or [], s, levels)
                    if chart_path:
                        ok = await send_photo(session, caption, chart_path)
                        print(f"[INFO] alert sent for {s}, send_photo ok={ok}")
                    else:
                        print("[WARN] chart not generated, sending text only")
                        await send_text(session, caption)
                except Exception as e:
                    print("[ERROR] while sending alert for", s, e)
                    await send_text(session, caption)
            elapsed = time.time() - start
            sleep_for = max(0, POLL_INTERVAL - elapsed)
            print(f"[TRACE] cycle complete, sleeping {sleep_for}s")
            await asyncio.sleep(sleep_for)

def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
        return
    asyncio.run(task_loop())

if __name__ == "__main__":
    main()
