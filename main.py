# main.py - Phase 4.8
# Multiframe analysis + OpenAI + Confidence scoring (includes OI jump & L/S trend) + Strong-alerts only + charts + Entry/SL/TP + cooldowns

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
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 1800))   # seconds
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", 3600))     # seconds
CONFIDENCE_MIN = int(os.getenv("CONFIDENCE_MIN", 60))   # require >=60% for STRONG alerts

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")             # optional
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=50"
OI_URL = "https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
LONGSHORT_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=30m&limit=1"

# ---------- Telegram helpers ----------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Missing TELEGRAM_BOT_TOKEN/CHAT_ID")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"Markdown"}, timeout=15) as r:
            txt = await r.text()
            print(f"[DEBUG] sendMessage status={r.status}")
    except Exception as e:
        print("[ERROR] send_text exception:", repr(e))

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Missing TELEGRAM_BOT_TOKEN/CHAT_ID")
        return False
    if not path or not os.path.exists(path):
        print("[ERROR] image missing:", path)
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
                print(f"[DEBUG] sendPhoto status={resp.status}")
                return resp.status == 200
    except Exception as e:
        print("[ERROR] send_photo exception:", repr(e))
        return False
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass

# ---------- Fetch helpers ----------
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
        times = [int(x[0]) // 1000 for x in raw]  # openTime ms -> s
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

# ---------- Local bias ----------
def simple_tf_bias(candles, lookback=10, threshold_pct=0.002):
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

# ---------- plot & trade ----------
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
        if times and len(times) == len(candles):
            dates = [datetime.utcfromtimestamp(t) for t in times]
        else:
            dates = [datetime.utcfromtimestamp(int(time.time()) - (len(candles)-i)*60*30) for i in range(len(candles))]
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

# ---------- OpenAI analysis (optional) ----------
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
        "Now analyze:\n" + "\n".join(parts)
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

# ---------- OI/LR feature helper ----------
def oi_ls_features(oi_now, oi_prev, ls_now, ls_prev):
    """
    returns dict with:
      - oi_change_pct
      - oi_spike_level: "none"/"medium"/"big"
      - ls_change
      - ls_note: "crowded_long"/"crowded_short"/"balanced"
    """
    out = {"oi_change_pct": None, "oi_spike_level": "none", "ls_change": None, "ls_note": "unknown"}
    try:
        if oi_prev and oi_now:
            pct = (oi_now - oi_prev) / oi_prev * 100.0
            out["oi_change_pct"] = pct
            if pct >= 20:
                out["oi_spike_level"] = "big"
            elif pct >= 10:
                out["oi_spike_level"] = "medium"
    except:
        pass
    try:
        if ls_prev is not None and ls_now is not None:
            out["ls_change"] = ls_now - ls_prev
            if ls_now > 2.0:
                out["ls_note"] = "crowded_long"
            elif ls_now < 0.5:
                out["ls_note"] = "crowded_short"
            else:
                out["ls_note"] = "balanced"
        else:
            out["ls_note"] = "unknown"
    except:
        out["ls_note"] = "unknown"
    return out

# ---------- CONFIDENCE logic (updated to include OI & L/S trend) ----------
def compute_confidence(local_biases, openai_bias, history, price, volume, long_short_ratio, prev_volume=None, prev_oi=None, prev_ls=None, prev_price=None, oi=None, symbol=None):
    """
    Returns (score:int 0-100, breakdown:dict)
    Weights:
      - local agreement (0-40)
      - openai confirmation (0-25)
      - long/short ratio directional strength (0-15)
      - recent volume spike (0-10)
      - history consistency (0-10)
      - OI spike gives additional bonus if aligning with price+bias
    """
    # candidate bias selection: prefer openai if present else local majority
    candidate = None
    if openai_bias in ("BUY","SELL"):
        candidate = openai_bias
    else:
        # pick majority among the three local tfs
        if local_biases:
            vals = list(local_biases.values())
            # fallback: if tie -> NEUTRAL
            candidate_counts = {}
            for v in vals:
                candidate_counts[v] = candidate_counts.get(v,0) + 1
            # remove NEUTRAL when possible
            sorted_items = sorted(candidate_counts.items(), key=lambda x: (-x[1], x[0]))
            candidate = sorted_items[0][0] if sorted_items else None
            if candidate == "NEUTRAL":
                # prefer any BUY/SELL if present in vals
                for v in vals:
                    if v in ("BUY","SELL"):
                        candidate = v
                        break

    if not candidate or candidate == "NEUTRAL":
        return 0, {"note":"no candidate bias"}

    # local agreement
    agrees = sum(1 for v in local_biases.values() if v == candidate)
    local_score = int((agrees/3.0) * 40)  # scale to 0..40

    # openai score
    openai_score = 25 if openai_bias == candidate else 0

    # long/short ratio contribution
    ls_score = 0
    if long_short_ratio is not None:
        diff = abs(long_short_ratio - 1.0)
        # if candidate BUY and ratio>1 => supportive
        if candidate == "BUY" and long_short_ratio > 1.0:
            # scale: diff 0..1 -> 0..15
            ls_score = min(15, int(diff * 15))
        if candidate == "SELL" and long_short_ratio < 1.0:
            ls_score = min(15, int(diff * 15))
        # penalize crowded same-side extremes slightly (encourage caution)
        if long_short_ratio > 2.0 and candidate == "BUY":
            # crowded long: reduce effective ls_score (risky)
            ls_score = max(0, ls_score - 5)
        if long_short_ratio < 0.5 and candidate == "SELL":
            ls_score = max(0, ls_score - 5)

    # volume spike
    vol_score = 0
    try:
        if prev_volume and volume and volume > 1.2 * prev_volume:
            vol_score = 10
        elif prev_volume and volume and volume > 1.05 * prev_volume:
            vol_score = 4
    except:
        vol_score = 0

    # history consistency
    hist_score = 0
    if len(history) >= 3 and history[-3:] == [history[-1]]*3 and history[-1] == candidate:
        hist_score = 10

    # OI spike and L/S trend features
    oi_bonus = 0
    oi_feats = oi_ls_features(oi, prev_oi, long_short_ratio, prev_ls)
    # if OI spike is big/medium and price moves in candidate direction -> bonus
    try:
        if oi_feats.get("oi_spike_level") == "big":
            # need price movement direction: if prev_price exists, compare
            if prev_price is not None and price is not None:
                if candidate == "BUY" and price > prev_price:
                    oi_bonus += 10
                if candidate == "SELL" and price < prev_price:
                    oi_bonus += 10
            else:
                oi_bonus += 6
        elif oi_feats.get("oi_spike_level") == "medium":
            if prev_price is not None and price is not None:
                if candidate == "BUY" and price > prev_price:
                    oi_bonus += 5
                if candidate == "SELL" and price < prev_price:
                    oi_bonus += 5
            else:
                oi_bonus += 2
        # L/S change positive in direction of candidate adds small bonus
        if oi_feats.get("ls_change") is not None:
            lsch = oi_feats.get("ls_change")
            if candidate == "BUY" and lsch > 0.05:
                oi_bonus += 3
            if candidate == "SELL" and lsch < -0.05:
                oi_bonus += 3
    except:
        oi_bonus = 0

    # sum scores
    score = local_score + openai_score + ls_score + vol_score + hist_score + oi_bonus
    if score > 100: score = 100
    breakdown = {
        "candidate": candidate,
        "local_agree": local_score,
        "openai": openai_score,
        "ls": ls_score,
        "vol": vol_score,
        "history": hist_score,
        "oi_bonus": oi_bonus,
        "total": score,
        "oi_feats": oi_feats
    }
    return score, breakdown

# ---------- decide_strong (same rule, but final gating by confidence) ----------
def decide_strong(local_biases, openai_bias, history):
    if len(history) >= 3 and history[-3:] == [history[-1]]*3 and history[-1] in ("BUY","SELL"):
        return True, history[-1], "3-cycle confirmation"
    if openai_bias in ("BUY","SELL"):
        agrees = sum(1 for v in local_biases.values() if v == openai_bias)
        if agrees >= 2:
            return True, openai_bias, f"openai+local majority ({agrees}/3)"
    return False, None, ""

# ---------- main loop ----------
async def main_loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session, "*Bot online — Phase 4.8 (OI+L/S integrated confidence)*")
        history = {s: [] for s in SYMBOLS}
        prev_vol = {s: None for s in SYMBOLS}
        prev_oi = {s: None for s in SYMBOLS}
        prev_ls = {s: None for s in SYMBOLS}
        prev_price = {s: None for s in SYMBOLS}
        cooldowns = {s: 0 for s in SYMBOLS}

        while True:
            start = time.time()
            tasks = [fetch_symbol(session, s) for s in SYMBOLS]
            results = await asyncio.gather(*tasks)
            market = {s:r for s,r in zip(SYMBOLS, results)}

            # compute local biases
            local = {}
            for s,d in market.items():
                local[s] = {
                    "30m": simple_tf_bias(d.get("candles_30m", [])),
                    "1h": simple_tf_bias(d.get("candles_1h", [])),
                    "4h": simple_tf_bias(d.get("candles_4h", []))
                }

            # openai analysis (optional)
            openai_out = await openai_analysis(market) if client else None

            # cycle summary
            summary_lines = [f"Snapshot (UTC {datetime.utcnow().strftime('%H:%M')})"]
            for s in SYMBOLS:
                d = market.get(s, {})
                summary_lines.append(f"{s}: {d.get('price','NA')} vol={d.get('volume','NA')}")
            await send_text(session, "🧠 " + "\n".join(summary_lines))

            # evaluate each symbol
            for s in SYMBOLS:
                d = market.get(s, {})
                loc = local.get(s, {})
                # majority local
                vals = list(loc.values()) if loc else []
                majority = "NEUTRAL"
                if vals:
                    # pick most common, tie breaks by pref BUY/SELL presence
                    counts = {}
                    for v in vals: counts[v] = counts.get(v,0) + 1
                    sorted_items = sorted(counts.items(), key=lambda x:(-x[1], x[0]))
                    majority = sorted_items[0][0]
                    if majority == "NEUTRAL":
                        for v in vals:
                            if v in ("BUY","SELL"):
                                majority = v
                                break

                history[s].append(majority)
                if len(history[s]) > 40: history[s].pop(0)

                openai_bias = None
                openai_reason = ""
                if openai_out and s in openai_out:
                    openai_bias = openai_out[s]["bias"]
                    openai_reason = openai_out[s].get("reason","")

                # basic potential strong rule
                potential, pot_bias, pot_note = decide_strong(loc, openai_bias, history[s])

                # compute confidence (pass prev values)
                conf_score, conf_breakdown = compute_confidence(
                    local_biases=loc,
                    openai_bias=openai_bias,
                    history=history[s],
                    price=d.get("price"),
                    volume=d.get("volume"),
                    long_short_ratio=d.get("long_short_ratio"),
                    prev_volume=prev_vol.get(s),
                    prev_oi=prev_oi.get(s),
                    prev_ls=prev_ls.get(s),
                    prev_price=prev_price.get(s),
                    oi=d.get("oi"),
                    symbol=s
                )

                # update prev trackers for next cycle
                prev_vol[s] = d.get("volume")
                prev_oi[s] = d.get("oi")
                prev_ls[s] = d.get("long_short_ratio")
                prev_price[s] = d.get("price")

                # final decision: need potential True AND conf >= threshold
                if potential and pot_bias:
                    if conf_score >= CONFIDENCE_MIN:
                        now_ts = time.time()
                        if now_ts < cooldowns[s]:
                            print(f"[DEBUG] cooldown active for {s}")
                        else:
                            levels = calc_levels_impl(d.get("candles_30m", []))
                            tl = compute_trade_levels(d.get("price"), levels, pot_bias)
                            caption = f"🚨 STRONG {pot_bias}: {s}\nTFs(local): 30m={loc.get('30m')},1h={loc.get('1h')},4h={loc.get('4h')}\nReason(openai): {openai_reason or 'N/A'}\nConfirm: {pot_note}\nConfidence: {conf_score}%"
                            # add a short numeric breakdown
                            caption += f"\nBreakdown: local={conf_breakdown['local_agree']}/40 openai={conf_breakdown['openai']}/25 ls={conf_breakdown['ls']}/15 vol={conf_breakdown['vol']}/10 hist={conf_breakdown['history']}/10 oi_bonus={conf_breakdown['oi_bonus']}"
                            if tl:
                                caption += f"\nEntry:{tl['entry']:.6f} SL:{tl['sl']:.6f} TP1:{tl['tp1']:.6f} TP2:{tl['tp2']:.6f}"
                            chart_path = plot_candles(d.get("times_30m", None), d.get("candles_30m", []), levels, s)
                            if chart_path:
                                ok = await send_photo(session, caption, chart_path)
                                print(f"[DEBUG] sent STRONG alert for {s}, ok={ok}, conf={conf_score}")
                            else:
                                await send_text(session, caption)
                            cooldowns[s] = now_ts + COOLDOWN_SEC
                    else:
                        print(f"[DEBUG] potential strong for {s} but confidence too low ({conf_score} < {CONFIDENCE_MIN})")
                        # optional: send small low-conf note (commented)
                        # await send_text(session, f"⚠️ {s} potential {pot_bias} but low confidence {conf_score}%")
                else:
                    # send informative chart for top symbols
                    if s in ("BTCUSDT","ETHUSDT","SOLUSDT"):
                        levels = calc_levels_impl(d.get("candles_30m", []))
                        tl = compute_trade_levels(d.get("price"), levels, majority if majority in ("BUY","SELL") else None)
                        caption = f"ℹ️ {s} {majority} (local majority) · OpenAI:{openai_bias or 'N/A'} · Conf:{conf_score}%"
                        if tl:
                            caption += f"\nEntry:{tl['entry']:.6f} SL:{tl['sl']:.6f} TP1:{tl['tp1']:.6f}"
                        chart_path = plot_candles(d.get("times_30m", None), d.get("candles_30m", []), levels, s)
                        if chart_path:
                            await send_photo(session, caption, chart_path)
                        else:
                            await send_text(session, caption)

            elapsed = time.time() - start
            to_sleep = max(0, POLL_INTERVAL - elapsed)
            print(f"[DEBUG] cycle done. sleeping {to_sleep} sec.")
            await asyncio.sleep(to_sleep)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("Interrupted")
