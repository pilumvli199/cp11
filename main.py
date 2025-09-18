#!/usr/bin/env python3
# main.py - Price-Action Bot (EMA9/20) with multimodal file-upload to OpenAI (gpt-4o-mini)
# Behavior:
# 1) Fetch Binance 30m candles (100)
# 2) Build 100-candle PNG chart with EMA9/20, price-action overlays
# 3) Upload PNG to OpenAI files endpoint (multipart). If fails -> fallback to base64 embed.
# 4) Send structured prompt + reference to uploaded file (file_id) + small base64 thumbnail
# 5) Parse GPT response for SYMBOL - ACTION - ENTRY/SL/TP - CONF and send Telegram message + PNG

import os
import re
import asyncio
import aiohttp
import traceback
import numpy as np
import base64
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple
import json
from io import BytesIO
from PIL import Image  # Pillow must be installed

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

RSI_PERIOD = 14
EMA_SHORT = 9
EMA_LONG = 20
VOLUME_MULTIPLIER = 2.0
MIN_CANDLES_FOR_ANALYSIS = 30
LOOKBACK_PERIOD = 100

price_history: Dict[str, List[Dict]] = {}
last_signals: Dict[str, datetime] = {}
performance_tracking: List[Dict] = []

# OpenAI client (used for non-file chat calls; file upload done via direct HTTP below)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=100"
ORDER_BOOK_URL = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
OPENAI_FILES_ENDPOINT = "https://api.openai.com/v1/files"

# ---------------- utils ----------------
def fmt_price(p: Optional[float]) -> str:
    if p is None:
        return "N/A"
    try:
        v = float(p)
    except:
        return str(p)
    return f"{v:.6f}" if abs(v) < 1 else f"{v:.2f}"

def fmt_decimal(val, s=6, l=2) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
    except:
        return str(val)
    return f"{v:.{s}f}" if abs(v) < 1 else f"{v:.{l}f}"

# ---------------- price-action helpers ----------------
def calculate_rsi(prices: List[float], period: int = RSI_PERIOD) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def ema_series(prices: List[float], span: int) -> List[float]:
    if not prices:
        return []
    alpha = 2 / (span + 1)
    out = [prices[0]]
    for p in prices[1:]:
        out.append(alpha * p + (1 - alpha) * out[-1])
    return out

def calculate_ema9_20(prices: List[float]) -> Tuple[Optional[float], Optional[float], List[float], List[float]]:
    if len(prices) < 2:
        return None, None, [], []
    e9 = ema_series(prices, EMA_SHORT)
    e20 = ema_series(prices, EMA_LONG)
    return round(e9[-1], 6), round(e20[-1], 6), e9, e20

# simplified PA pattern detection & key levels (same as earlier patterns)
def detect_price_action_patterns(candles: List[List[float]]) -> Dict[str, bool]:
    patterns = {}
    if len(candles) < 5:
        return patterns
    recent = candles[-5:]
    closes = [c[3] for c in recent]
    highs = [c[1] for c in recent]
    lows = [c[2] for c in recent]
    opens = [c[0] for c in recent]
    if len(candles) > 20:
        prev_high = max([c[1] for c in candles[-21:-1]])
        prev_low = min([c[2] for c in candles[-21:-1]])
        if closes[-1] > prev_high:
            patterns['resistance_break'] = True
        if closes[-1] < prev_low:
            patterns['support_break'] = True
    if len(recent) >= 2:
        o, h, l, c = recent[-1]
        po, ph, pl, pc = recent[-2]
        body = abs(c - o)
        prev_body = abs(pc - po)
        if c > o and pc < po and c > po and o < pc and body > prev_body * 1.2:
            patterns['bullish_engulfing'] = True
        if c < o and pc > po and c < po and o > pc and body > prev_body * 1.2:
            patterns['bearish_engulfing'] = True
        upper_wick = h - max(o,c)
        lower_wick = min(o,c) - l
        if upper_wick > body * 2 and lower_wick < body * 0.5 and c < o:
            patterns['shooting_star'] = True
        if lower_wick > body * 2 and upper_wick < body * 0.5 and c > o:
            patterns['hammer'] = True
    return patterns

def calculate_support_resistance(candles: List[List[float]]) -> Dict[str, float]:
    levels = {}
    if len(candles) < 20:
        return levels
    highs = [c[1] for c in candles]; lows = [c[2] for c in candles]
    swing_highs = []; swing_lows = []
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            swing_lows.append(lows[i])
    if swing_highs:
        levels['resistance_1'] = max(swing_highs[-3:]) if len(swing_highs) >= 3 else swing_highs[-1]
    if swing_lows:
        levels['support_1'] = min(swing_lows[-3:]) if len(swing_lows) >= 3 else swing_lows[-1]
    closes = [c[3] for c in candles]; cp = closes[-1]
    if levels:
        if 'resistance_1' in levels:
            levels['distance_to_resistance'] = ((levels['resistance_1'] - cp) / cp) * 100
        if 'support_1' in levels:
            levels['distance_to_support'] = ((cp - levels['support_1']) / cp) * 100
    return levels

# ---------------- Fetching ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=25) as r:
            if r.status != 200:
                txt = await r.text()
                print(f"fetch_json {url} returned {r.status}: {txt[:200]}")
                return None
            return await r.json()
    except Exception as e:
        print("fetch_json exception for", url, e)
        return None

async def fetch_enhanced_data(session, symbol):
    ticker_task = fetch_json(session, TICKER_URL.format(symbol=symbol))
    candle_task = fetch_json(session, CANDLE_URL.format(symbol=symbol))
    orderbook_task = fetch_json(session, ORDER_BOOK_URL.format(symbol=symbol))
    ticker, candles, orderbook = await asyncio.gather(ticker_task, candle_task, orderbook_task)
    out = {}
    if ticker:
        try:
            out["price"] = float(ticker.get("lastPrice", 0))
            out["volume"] = float(ticker.get("volume", 0))
            out["price_change_24h"] = float(ticker.get("priceChangePercent", 0))
            out["high_24h"] = float(ticker.get("highPrice", 0))
            out["low_24h"] = float(ticker.get("lowPrice", 0))
            out["quote_volume"] = float(ticker.get("quoteVolume", 0))
        except Exception as e:
            print("Error processing ticker:", e)
            out["price"] = None
    if isinstance(candles, list) and len(candles) >= MIN_CANDLES_FOR_ANALYSIS:
        try:
            parsed_candles = []; times = []; volumes = []
            for x in candles:
                open_price = float(x[1]); high = float(x[2]); low = float(x[3]); close = float(x[4]); volume = float(x[5])
                timestamp = int(x[0]) // 1000
                parsed_candles.append([open_price, high, low, close])
                times.append(timestamp); volumes.append(volume)
            out["candles"] = parsed_candles; out["times"] = times; out["volumes"] = volumes
            closes = [c[3] for c in parsed_candles]
            out["rsi"] = calculate_rsi(closes, RSI_PERIOD)
            out["ema9"], out["ema20"], out["ema9_series"], out["ema20_series"] = calculate_ema9_20(closes)
            out["patterns"] = detect_price_action_patterns(parsed_candles)
            out["key_levels"] = calculate_support_resistance(parsed_candles)
            # order book derived
            if orderbook:
                bids = [(float(x[0]), float(x[1])) for x in orderbook.get("bids", [])]
                asks = [(float(x[0]), float(x[1])) for x in orderbook.get("asks", [])]
                if bids and asks:
                    out["bid"] = bids[0][0]; out["ask"] = asks[0][0]
                    avg_bid = np.mean([b[1] for b in bids[:5]]) if bids else 0
                    avg_ask = np.mean([a[1] for a in asks[:5]]) if asks else 0
                    sig_bids = [x for x in bids if x[1] > avg_bid*2]
                    sig_asks = [x for x in asks if x[1] > avg_ask*2]
                    out["ob_support"] = sig_bids[0][0] if sig_bids else None
                    out["ob_resistance"] = sig_asks[0][0] if sig_asks else None
        except Exception as e:
            print("Candle processing error:", e)
            traceback.print_exc()
    if out.get("price") is not None:
        price_history.setdefault(symbol, []).append({"price": out["price"], "timestamp": datetime.now()})
        if len(price_history[symbol]) > 200:
            price_history[symbol] = price_history[symbol][-200:]
    return out

# ---------------- Charting ----------------
def enhanced_plot_chart(times, candles, symbol, market_data, out_path=None, img_maxsize=(1000,800)):
    if not times or not candles or len(times) != len(candles) or len(candles) < 10:
        raise ValueError("Insufficient data for plotting")
    dates = [datetime.utcfromtimestamp(int(t)) for t in times]
    closes = [c[3] for c in candles]; highs = [c[1] for c in candles]; lows = [c[2] for c in candles]
    x = date2num(dates)
    fig = plt.figure(figsize=(14,9), dpi=120, constrained_layout=True)
    gs = fig.add_gridspec(3,1, height_ratios=[3,1,1], hspace=0.25)
    ax_price = fig.add_subplot(gs[0]); ax_vol = fig.add_subplot(gs[1]); ax_rsi = fig.add_subplot(gs[2])
    width = 0.6*(x[1]-x[0]) if len(x)>1 else 0.4
    for xi, candle in zip(x, candles):
        o,h,l,c = candle
        color = "#006600" if c>=o else "#660000"
        ax_price.vlines(xi, l, h, color=color, linewidth=1.1, alpha=0.9)
        rect_h = abs(c-o) if abs(c-o)>0.0001 else 0.0001
        rect = plt.Rectangle((xi-width/2, min(o,c)), width, rect_h, facecolor=color, edgecolor=color, alpha=0.9)
        ax_price.add_patch(rect)
    # EMAs
    e9 = market_data.get("ema9_series", []); e20 = market_data.get("ema20_series", [])
    if e9 and e20 and len(e9) == len(x):
        ax_price.plot(x, e9, linewidth=1.2, label=f"EMA{EMA_SHORT}")
        ax_price.plot(x, e20, linewidth=1.2, label=f"EMA{EMA_LONG}")
    # key levels
    kl = market_data.get("key_levels", {})
    if kl.get("support_1"):
        ax_price.axhline(kl["support_1"], color="blue", linestyle="--", linewidth=1.4, label=f"S1 {fmt_price(kl['support_1'])}")
    if kl.get("resistance_1"):
        ax_price.axhline(kl["resistance_1"], color="red", linestyle="--", linewidth=1.4, label=f"R1 {fmt_price(kl['resistance_1'])}")
    # volume
    vols = market_data.get("volumes", [])
    if vols and len(vols) == len(x):
        colors = []
        for i in range(len(candles)):
            if i==0: colors.append("gray")
            elif candles[i][3] >= candles[i-1][3]: colors.append("green")
            else: colors.append("red")
        ax_vol.bar(x, vols, width=width, color=colors, alpha=0.7)
        ax_vol.set_ylabel("Vol")
    # rsi
    if len(closes) >= RSI_PERIOD + 5:
        rsi_vals = []
        for i in range(RSI_PERIOD, len(closes)):
            r = calculate_rsi(closes[:i+1], RSI_PERIOD)
            if r is not None:
                rsi_vals.append(r)
        if rsi_vals:
            rx = x[-len(rsi_vals):]
            ax_rsi.plot(rx, rsi_vals, linewidth=1.1)
            ax_rsi.axhline(70, linestyle="--", alpha=0.6); ax_rsi.axhline(30, linestyle="--", alpha=0.6)
            ax_rsi.set_ylim(0,100)
    current_price = closes[-1]
    title = f"{symbol} | Price: ${fmt_price(current_price)} | EMA{EMA_SHORT}: {fmt_decimal(market_data.get('ema9'))} EMA{EMA_LONG}: {fmt_decimal(market_data.get('ema20'))}"
    ax_price.set_title(title, fontsize=12, fontweight="bold")
    ax_price.legend(loc="upper left", fontsize="small")
    ax_price.grid(alpha=0.25)
    fig.autofmt_xdate()
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)
    # shrink with PIL & save to out_path or tmp.name
    final_path = out_path if out_path else tmp.name
    try:
        with Image.open(tmp.name) as im:
            im = im.convert("RGB")
            im.thumbnail(img_maxsize, Image.LANCZOS)
            im.save(final_path, format="PNG", optimize=True)
    except Exception:
        # if PIL fails, keep tmp
        final_path = tmp.name
    return final_path

# ---------------- OpenAI file upload (multipart) ----------------
async def upload_file_to_openai(session: aiohttp.ClientSession, file_path: str, purpose: str = "inputs") -> Optional[str]:
    """
    Upload file to OpenAI /v1/files endpoint using aiohttp multipart.
    Returns file id on success, else None.
    """
    if not OPENAI_API_KEY:
        print("No OPENAI_API_KEY set.")
        return None
    try:
        with open(file_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename=os.path.basename(file_path), content_type="image/png")
            data.add_field("purpose", purpose)
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            async with session.post(OPENAI_FILES_ENDPOINT, data=data, headers=headers, timeout=60) as resp:
                txt = await resp.text()
                if resp.status != 200 and resp.status != 201:
                    print(f"OpenAI file upload failed {resp.status}: {txt[:1000]}")
                    return None
                try:
                    j = await resp.json()
                    file_id = j.get("id") or j.get("file_id") or j.get("name")
                    print("Uploaded file, response keys:", list(j.keys()))
                    return file_id
                except Exception:
                    print("Upload returned non-json:", txt[:500])
                    return None
    except Exception as e:
        print("upload_file_to_openai exception:", e)
        return None

# ---------------- OpenAI analysis with file reference ----------------
async def ask_openai_with_file(session: aiohttp.ClientSession, market_summary_text: str, chart_path: Optional[str]) -> Optional[str]:
    """
    Try to upload the chart to OpenAI and include file reference in the prompt.
    If upload fails, fallback to small base64 embed.
    """
    # 1) Try upload
    file_id = None
    if chart_path and os.path.exists(chart_path):
        file_id = await upload_file_to_openai(session, chart_path, purpose="analysis")
    # 2) Build small thumbnail base64 (safer to include small image)
    thumb_b64 = None
    if chart_path and os.path.exists(chart_path):
        try:
            with Image.open(chart_path) as im:
                im = im.convert("RGB")
                im.thumbnail((600,400), Image.LANCZOS)
                bio = BytesIO()
                im.save(bio, format="PNG", optimize=True)
                thumb_b64 = base64.b64encode(bio.getvalue()).decode("ascii")
        except Exception as e:
            print("Thumbnail creation error:", e)
            thumb_b64 = None
    # 3) Build prompt: include file_id (if present) and small base64 (if available)
    prompt_system = "You are an expert price-action crypto trader. Analyze the provided market summary and chart (image provided as file or thumbnail). Provide high-probability trade setups in the required strict format."
    prompt_user = f"""MARKET SUMMARY (JSON):
{market_summary_text}

INSTRUCTIONS:
- Only return signals with CONF >= {int(SIGNAL_CONF_THRESHOLD)}%
- Output lines: SYMBOL - ACTION - ENTRY: <price> - SL: <price> - TP: <price> - REASON: <<=50 words> - CONF: <{int(SIGNAL_CONF_THRESHOLD)}-100>%
- Use EMA{EMA_SHORT}/{EMA_LONG} for momentum context. Min R:R 1.5:1.
- If no valid signals, reply exactly: NO_SIGNAL

"""
    if file_id:
        prompt_user += f"\nNOTE: Chart uploaded to OpenAI files with id: {file_id}. Use the image content for visual analysis.\n"
    if thumb_b64:
        # include as small base64 block for backup
        prompt_user += "\n---BEGIN_THUMBNAIL_BASE64---\n"
        prompt_user += thumb_b64
        prompt_user += "\n---END_THUMBNAIL_BASE64---\n"
    # Call Chat Completions via OpenAI client if available, else fallback to direct HTTP
    try:
        loop = asyncio.get_running_loop()
        def call_model():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role":"system","content":prompt_system},
                    {"role":"user","content":prompt_user}
                ],
                max_tokens=1200,
                temperature=0.05
            )
        resp = await loop.run_in_executor(None, call_model)
        try:
            content = resp.choices[0].message.content
            print("OpenAI response preview:", (content or "")[:1000])
            return content.strip() if content else None
        except Exception:
            return str(resp)
    except Exception as e:
        print("OpenAI client chat call failed, falling back to HTTP (this will likely still use your client lib). Error:", e)
        # As a fallback: send HTTP POST to /v1/chat/completions (careful: depends on model availability)
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
            payload = {
                "model": OPENAI_MODEL,
                "messages":[{"role":"system","content":prompt_system},{"role":"user","content":prompt_user}],
                "max_tokens":1200,
                "temperature":0.05
            }
            async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=60) as r:
                txt = await r.text()
                if r.status != 200:
                    print("HTTP chat completions failed:", r.status, txt[:1000])
                    return None
                j = await r.json()
                # try to extract text
                try:
                    return j["choices"][0]["message"]["content"].strip()
                except Exception:
                    return str(j)
        except Exception as ee:
            print("HTTP fallback failed:", ee)
            return None

# ---------------- parse GPT signals ----------------
def enhanced_parse(text: Optional[str]):
    out = {}
    if not text:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper() == "NO_SIGNAL":
            continue
        if not any(k in line.upper() for k in ("BUY","SELL")):
            continue
        parts = [p.strip() for p in line.split(" - ")]
        if len(parts) < 3:
            continue
        symbol = parts[0].upper().replace(" ", "")
        action = parts[1].upper()
        if action not in ("BUY","SELL"):
            continue
        entry = sl = tp = None; reason = ""; conf = None
        remainder = " - ".join(parts[2:])
        m_entry = re.search(r'ENTRY\s*[:=]\s*([0-9\.]+)', remainder, flags=re.I)
        m_sl = re.search(r'\bSL\b\s*[:=]\s*([0-9\.]+)', remainder, flags=re.I)
        m_tp = re.search(r'\bTP\b\s*[:=]\s*([0-9\.]+)', remainder, flags=re.I)
        m_conf = re.search(r'CONF(?:IDENCE)?\s*[:=]?\s*(\d{2,3})', remainder, flags=re.I)
        m_reason = re.search(r'REASON\s*[:=]\s*(.+?)(?:\s*-\s*CONF|$)', remainder, flags=re.I)
        if m_entry: entry = float(m_entry.group(1))
        if m_sl: sl = float(m_sl.group(1))
        if m_tp: tp = float(m_tp.group(1))
        if m_conf: conf = int(m_conf.group(1))
        if m_reason: reason = m_reason.group(1).strip()
        if not all([entry, sl, tp, conf]):
            print(f"Incomplete signal parsed for {symbol}: {entry},{sl},{tp},{conf}")
            continue
        if conf < SIGNAL_CONF_THRESHOLD:
            print(f"Signal below threshold: {conf}% < {SIGNAL_CONF_THRESHOLD}%")
            continue
        if action == "BUY":
            risk = entry - sl; reward = tp - entry
        else:
            risk = sl - entry; reward = entry - tp
        if risk <= 0 or reward <= 0:
            print("Invalid R/R skip")
            continue
        rr = reward / risk
        if rr < 1.5:
            print(f"Poor R:R {rr:.2f} skip")
            continue
        if symbol in last_signals and datetime.now() - last_signals[symbol] < timedelta(hours=2):
            print(f"Recent signal exists for {symbol}, skipping")
            continue
        out[symbol] = {"action":action,"entry":entry,"sl":sl,"tp":tp,"reason":reason,"confidence":conf,"risk_reward":round(rr,2),"timestamp":datetime.now()}
        last_signals[symbol] = datetime.now()
    return out

# ---------------- Telegram ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("TG not configured; would send text:", text[:300])
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"Markdown", "disable_web_page_preview": True}, timeout=30) as r:
            if r.status != 200:
                txt = await r.text()
                print("Telegram send_text failed:", r.status, txt[:300])
    except Exception as e:
        print("send_text error:", e)

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("TG not configured; would send photo with caption:", caption[:200])
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("chat_id", str(TELEGRAM_CHAT_ID))
            data.add_field("caption", caption)
            data.add_field("parse_mode", "Markdown")
            data.add_field("photo", f, filename=os.path.basename(path), content_type="image/png")
            async with session.post(url, data=data, timeout=90) as r:
                if r.status != 200:
                    txt = await r.text()
                    print("Telegram send_photo failed:", r.status, txt[:500])
    except Exception as e:
        print("send_photo error:", e)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# ---------------- Main loop ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as session:
        startup = f"🤖 Price-Action Bot (EMA{EMA_SHORT}/{EMA_LONG}) online. Symbols: {len(SYMBOLS)}, Poll: {POLL_INTERVAL}s"
        print(startup)
        await send_text(session, startup)
        iteration = 0
        while True:
            iteration += 1
            start_time = datetime.now()
            print(f"\nITERATION {iteration} @ {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                tasks = [fetch_enhanced_data(session, s) for s in SYMBOLS]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                market = {}
                for s, res in zip(SYMBOLS, results):
                    if isinstance(res, Exception):
                        print(f"Fetch error {s}: {res}")
                        continue
                    if res and res.get("price") is not None:
                        market[s] = res
                        print(f"✅ {s}: ${fmt_price(res['price'])}")
                    else:
                        print(f"⚠️ No data for {s}")
                if not market:
                    print("No market data; sleeping")
                    await asyncio.sleep(min(120, POLL_INTERVAL))
                    continue
                signals_found = {}
                for symbol, data in market.items():
                    summary = {
                        "symbol": symbol,
                        "price": data.get("price"),
                        "ema9": data.get("ema9"),
                        "ema20": data.get("ema20"),
                        "patterns": data.get("patterns"),
                        "key_levels": data.get("key_levels"),
                        "volume_analysis": data.get("volume_analysis"),
                        "ob_support": data.get("ob_support"),
                        "ob_resistance": data.get("ob_resistance"),
                        "candles_count": len(data.get("candles", []))
                    }
                    summary_text = json.dumps(summary, default=str, indent=2)
                    chart_path = None
                    if data.get("candles") and data.get("times"):
                        try:
                            chart_path = enhanced_plot_chart(data["times"], data["candles"], symbol, data, img_maxsize=(1000,800))
                        except Exception as e:
                            print("Chart error:", e)
                    # call OpenAI with file upload attempt
                    ai_resp = await ask_openai_with_file(session, summary_text, chart_path)
                    print(f"--- GPT raw for {symbol} ---\n{(ai_resp or '<empty>')[:3000]}\n--- end ---")
                    parsed = enhanced_parse(ai_resp or "")
                    for k,v in parsed.items():
                        signals_found[k] = {"signal": v, "chart": chart_path}
                        performance_tracking.append({
                            "symbol": k,
                            "action": v["action"],
                            "entry": v["entry"],
                            "sl": v["sl"],
                            "tp": v["tp"],
                            "confidence": v["confidence"],
                            "risk_reward": v["risk_reward"],
                            "timestamp": datetime.now(),
                            "reason": v.get("reason","")
                        })
                if signals_found:
                    print(f"🚨 Found {len(signals_found)} signals")
                    for sym, payload in signals_found.items():
                        sig = payload["signal"]; chart = payload.get("chart")
                        action = sig["action"]; entry = sig["entry"]; sl = sig["sl"]; tp = sig["tp"]
                        conf = sig["confidence"]; rr = sig["risk_reward"]; reason = sig.get("reason","")
                        emoji = "🟢" if action=="BUY" else "🔴"
                        potential = ((tp - entry)/entry)*100 if action=="BUY" else ((entry - tp)/entry)*100
                        riskpct = ((entry - sl)/entry)*100 if action=="BUY" else ((sl - entry)/entry)*100
                        caption = f"""{emoji} *SIGNAL* {emoji}

*{sym}* → *{action}*
Entry: `{fmt_price(entry)}`  SL: `{fmt_price(sl)}`  TP: `{fmt_price(tp)}`
Confidence: *{conf}%*  R:R: *1:{rr}*  Risk: {riskpct:.1f}%  Potential: +{potential:.1f}%

_Reason:_ {reason}
"""
                        if chart:
                            await send_photo(session, caption, chart)
                        else:
                            await send_text(session, caption)
                else:
                    print("No high-confidence signals this iteration.")
                if iteration % 12 == 0:
                    wins = sum(1 for p in performance_tracking if p.get("profit",0)>0)
                    print(f"Perf: total {len(performance_tracking)}, wins {wins}")
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"Iteration finished in {elapsed:.1f}s. Sleeping {POLL_INTERVAL}s")
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                print("Shutdown requested")
                await send_text(session, "Bot shutting down")
                break
            except Exception as e:
                print("MAIN LOOP ERROR:", e)
                traceback.print_exc()
                try:
                    await send_text(session, f"⚠️ Bot error: {str(e)[:200]}")
                except:
                    pass
                await asyncio.sleep(min(120, POLL_INTERVAL))

if __name__ == "__main__":
    print("Starting Price-Action Bot (EMA9/20) with multimodal file-upload (B)...")
    try:
        asyncio.run(enhanced_loop())
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as e:
        print("Fatal:", e)
        traceback.print_exc()
