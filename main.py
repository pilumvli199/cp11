# main.py - GPT-driven Crypto Bot with chart image + text alerts (30m TF, last up to 100 candles)
import os
import re
import asyncio
import aiohttp
import traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple

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
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 75.0))

# Analysis windows
RSI_PERIOD = 14
MA_SHORT = 7
MA_LONG = 21
VOLUME_MULTIPLIER = 1.5
MIN_CANDLES_FOR_ANALYSIS = 20   # minimal to process, but we will use up to 100
LOOKBACK_PERIOD = 100           # last up to 100 candles for analysis/chart

price_history: Dict[str, List[Dict]] = {}
signal_history: List[Dict] = []

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=100"
ORDER_BOOK_URL = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit=10"

# ---------------- Helpers / Indicators ----------------
def to_float_from_text(s: str) -> Optional[float]:
    if not s:
        return None
    # remove currency symbols, commas, and stray text
    cleaned = re.sub(r'[^\d\.\-]', '', s.replace(',', ''))
    if cleaned in ("", ".", "-", "-."):
        return None
    try:
        return float(cleaned)
    except:
        return None

def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change); losses.append(0)
        else:
            gains.append(0); losses.append(abs(change))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def calculate_moving_averages(prices: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(prices) < MA_LONG:
        return None, None
    ma_short = sum(prices[-MA_SHORT:]) / MA_SHORT if len(prices) >= MA_SHORT else None
    ma_long = sum(prices[-MA_LONG:]) / MA_LONG
    return ma_short, ma_long

def detect_volume_spike(volumes: List[float]) -> bool:
    if len(volumes) < 10:
        return False
    avg_volume = sum(volumes[-10:-1]) / 9
    return volumes[-1] > (avg_volume * VOLUME_MULTIPLIER)

def enhanced_levels(candles, lookback=LOOKBACK_PERIOD):
    """Return primary_support, primary_resistance, secondary_support, secondary_resistance, mid_level"""
    if not candles or len(candles) < 6:
        return (None, None, None, None, None)
    arr = candles[-min(len(candles), lookback):]
    highs = [c[1] for c in arr]; lows = [c[2] for c in arr]; closes = [c[3] for c in arr]
    highs_sorted = sorted(highs, reverse=True); lows_sorted = sorted(lows)
    primary_resistance = sum(highs_sorted[:3]) / 3 if len(highs_sorted) >= 3 else None
    primary_support = sum(lows_sorted[:3]) / 3 if len(lows_sorted) >= 3 else None
    secondary_resistance = sum(highs_sorted[3:6]) / 3 if len(highs_sorted) >= 6 else None
    secondary_support = sum(lows_sorted[3:6]) / 3 if len(lows_sorted) >= 6 else None
    current_price = closes[-1]
    mid_level = (primary_resistance + primary_support) / 2 if primary_resistance and primary_support else current_price
    return primary_support, primary_resistance, secondary_support, secondary_resistance, mid_level

def detect_patterns(candles) -> Dict[str, bool]:
    if len(candles) < 5:
        return {}
    patterns = {}
    last5 = candles[-5:]
    last = last5[-1]
    body = abs(last[3] - last[0])
    rng = last[1] - last[2]
    patterns['doji'] = (body / rng) < 0.1 if rng > 0 else False
    try:
        lower_wick = (last[0] - last[2]) if last[0] > last[2] else (last[3] - last[2])
        upper_wick = last[1] - max(last[0], last[3])
    except Exception:
        lower_wick = upper_wick = 0
    patterns['hammer'] = (lower_wick > 2 * body) and (upper_wick < body) if body > 0 else False
    if len(last5) >= 2:
        prev, curr = last5[-2], last5[-1]
        patterns['bullish_engulfing'] = (curr[3] > curr[0]) and (prev[3] < prev[0]) and (curr[3] > prev[0]) and (curr[0] < prev[3])
        patterns['bearish_engulfing'] = (curr[3] < curr[0]) and (prev[3] > prev[0]) and (curr[3] < prev[0]) and (curr[0] > prev[3])
    return patterns

# ---------------- Fetching ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=15) as r:
            if r.status != 200:
                txt = await r.text() if r is not None else "<no body>"
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
        except Exception as e:
            print(f"Error processing ticker for {symbol}: {e}")
            out["price"] = None
            out["volume"] = None
    if isinstance(candles, list) and len(candles) >= MIN_CANDLES_FOR_ANALYSIS:
        try:
            out["candles"] = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in candles]
            out["times"] = [int(x[0]) // 1000 for x in candles]
            out["volumes"] = [float(x[5]) for x in candles]
            closes = [float(x[4]) for x in candles]
            out["rsi"] = calculate_rsi(closes, RSI_PERIOD)
            out["ma_short"], out["ma_long"] = calculate_moving_averages(closes)
            out["volume_spike"] = detect_volume_spike(out["volumes"])
            out["patterns"] = detect_patterns(out["candles"])
        except Exception as e:
            print("Enhanced candle processing error for", symbol, e)
            out["candles"] = None
    if orderbook:
        try:
            bids = [(float(x[0]), float(x[1])) for x in orderbook.get("bids", [])]
            asks = [(float(x[0]), float(x[1])) for x in orderbook.get("asks", [])]
            if bids and asks:
                out["bid"] = bids[0][0]
                out["ask"] = asks[0][0]
                out["spread"] = asks[0][0] - bids[0][0]
                total_bid_volume = sum(x[1] for x in bids[:5])
                total_ask_volume = sum(x[1] for x in asks[:5])
                out["buy_pressure"] = (total_bid_volume / total_ask_volume) if total_ask_volume > 0 else 0
        except Exception as e:
            print(f"Order book processing error for {symbol}: {e}")
    if symbol not in price_history:
        price_history[symbol] = []
    if out.get("price") is not None:
        price_history[symbol].append({"price": out["price"], "timestamp": datetime.now(), "volume": out.get("volume", 0)})
        if len(price_history[symbol]) > 100:
            price_history[symbol] = price_history[symbol][-100:]
    return out

# ---------------- OpenAI Analysis (ask for ENTRY/SL/TP) ----------------
async def enhanced_analyze_openai(market):
    if not client:
        print("No OpenAI client configured.")
        return None
    analysis_parts = []
    for symbol, data in market.items():
        if not data.get("price"):
            continue
        # Include summary + last up to 100 candles (compact string)
        s = f"\n{symbol}: price={data['price']}, rsi={data.get('rsi')}, ma_short={data.get('ma_short')}, ma_long={data.get('ma_long')}, vol_spike={data.get('volume_spike')}"
        if data.get("candles"):
            # include last up to 100 closes (compact)
            closes = [c[3] for c in data["candles"]][-100:]
            s += f", last_closes_len={len(closes)}"
        analysis_parts.append(s)
    if not analysis_parts:
        print("No market data available for enhanced analysis.")
        return None

    prompt = f"""You are an expert crypto trading analyst. Use the provided market data (30-minute timeframe, up to last 100 candles) to identify high-probability trades.

For each strong signal provide:
- ACTION: BUY or SELL
- ENTRY price (near market or breakout)
- STOPLOSS (recent swing high/low or support/resistance)
- TARGET (use at least 1:2 risk:reward; give a numeric TP)
- REASON: brief technical justification (patterns, S/R, MA, RSI, volume, orderbook)
- CONF: confidence percent (70-100)

Output strict one-line-per-signal format exactly like:
SYMBOL - ACTION - ENTRY: <price> - SL: <price> - TP: <price> - REASON: <text> - CONF: XX%

Do NOT output neutral signals. Only output signals with clear SL/TP and CONF >= 70.

MARKET SUMMARY:
{"".join(analysis_parts)}

Remember: be concise, numeric prices only for ENTRY/SL/TP (no ranges), confidence as integer percent.
"""
    try:
        loop = asyncio.get_running_loop()
        def call_model():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=900,
                temperature=0.0
            )
        resp = await loop.run_in_executor(None, call_model)
        # extract text robustly
        try:
            choice = resp.choices[0]
            content = choice.message.content if hasattr(choice, "message") else getattr(choice, "text", None)
            if content is None:
                content = str(resp)
            # debug print
            print("OpenAI response:\n", content[:2000])
            return content.strip()
        except Exception:
            return str(resp)
    except Exception as e:
        print("Enhanced OpenAI call failed:", e)
        traceback.print_exc()
        return None

# ---------------- Plotting (candles -> PNG) ----------------
def enhanced_plot_chart(times, candles, symbol, market_data):
    if not times or not candles or len(times) != len(candles) or len(candles) < 10:
        raise ValueError("Insufficient data for enhanced plotting")
    dates = [datetime.utcfromtimestamp(int(t)) for t in times]
    closes = [c[3] for c in candles]
    x = date2num(dates)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=100, gridspec_kw={'height_ratios': [3,1]})
    width = 0.6 * (x[1] - x[0]) if len(x) > 1 else 0.4
    for xi, candle in zip(x, candles):
        o, h, l, c = candle
        color = "green" if c >= o else "red"
        edge_color = "darkgreen" if c >= o else "darkred"
        ax1.vlines(xi, l, h, color=edge_color, linewidth=0.8)
        rect_height = abs(c - o) if abs(c - o) > 0.0001 else 0.0001
        rect = plt.Rectangle((xi - width/2, min(o, c)), width, rect_height,
                             facecolor=color, edgecolor=edge_color, alpha=0.8)
        ax1.add_patch(rect)
    sup1, res1, sup2, res2, mid = enhanced_levels(candles, lookback=LOOKBACK_PERIOD)
    if res1: ax1.axhline(res1, color="red", linestyle="--", alpha=0.7, label=f"Primary Resistance: {res1:.6f}" if res1 < 1 else f"Primary Resistance: {res1:.2f}")
    if sup1: ax1.axhline(sup1, color="blue", linestyle="--", alpha=0.7, label=f"Primary Support: {sup1:.6f}" if sup1 < 1 else f"Primary Support: {sup1:.2f}")
    if res2: ax1.axhline(res2, color="orange", linestyle=":", alpha=0.5, label=f"Secondary Resistance: {res2:.2f}")
    if sup2: ax1.axhline(sup2, color="cyan", linestyle=":", alpha=0.5, label=f"Secondary Support: {sup2:.2f}")
    # MAs
    if len(closes) >= MA_LONG:
        ma_short_values, ma_long_values = [], []
        for i in range(len(closes)):
            if i >= MA_SHORT - 1:
                ma_short_values.append(sum(closes[i-MA_SHORT+1:i+1]) / MA_SHORT)
            else: ma_short_values.append(None)
            if i >= MA_LONG - 1:
                ma_long_values.append(sum(closes[i-MA_LONG+1:i+1]) / MA_LONG)
            else: ma_long_values.append(None)
        valid_short = [(x[i], ma) for i, ma in enumerate(ma_short_values) if ma is not None]
        valid_long = [(x[i], ma) for i, ma in enumerate(ma_long_values) if ma is not None]
        if valid_short:
            x_short, y_short = zip(*valid_short)
            ax1.plot(x_short, y_short, color="purple", linewidth=1.5, alpha=0.8, label=f"MA{MA_SHORT}")
        if valid_long:
            x_long, y_long = zip(*valid_long)
            ax1.plot(x_long, y_long, color="brown", linewidth=1.5, alpha=0.8, label=f"MA{MA_LONG}")
    # RSI
    if market_data.get("candles") and len(market_data["candles"]) >= RSI_PERIOD + 5:
        rsi_values = []
        for i in range(RSI_PERIOD, len(closes)):
            rsi = calculate_rsi(closes[:i+1], RSI_PERIOD)
            rsi_values.append(rsi)
        if rsi_values:
            rsi_x = x[-len(rsi_values):]
            ax2.plot(rsi_x, rsi_values, linewidth=1.5)
            ax2.axhline(70, color="red", linestyle="--", alpha=0.7)
            ax2.axhline(30, color="green", linestyle="--", alpha=0.7)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("RSI", fontsize=10)
            ax2.grid(True, alpha=0.3)
    current_price = closes[-1]
    rsi_current = market_data.get("rsi", "N/A")
    if current_price < 1:
        price_str = f"{current_price:.6f}"
    else:
        price_str = f"{current_price:.2f}"
    ax1.set_title(f"{symbol} - Price: {price_str} | RSI: {rsi_current}", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize="small")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return tmp.name

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
        try: os.remove(path)
        except: pass
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
        try: os.remove(path)
        except Exception: pass

# ---------------- Parsing AI output (ENTRY/SL/TP) ----------------
def enhanced_parse(text):
    out = {}
    if not text:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or not any(k in line.upper() for k in ("BUY","SELL")):
            continue
        # expected format:
        # SYMBOL - ACTION - ENTRY: <price> - SL: <price> - TP: <price> - REASON: ... - CONF: XX%
        parts = [p.strip() for p in line.split(" - ")]
        if len(parts) < 3:
            continue
        symbol = parts[0].upper()
        action = parts[1].upper()
        entry = sl = tp = None
        reason = ""
        conf = None
        # join remaining and then search tokens
        remainder = " - ".join(parts[2:])
        # extract ENTRY, SL, TP using regex
        m_entry = re.search(r'ENTRY\s*[:=]\s*([0-9\.,]+)', remainder, flags=re.I)
        m_sl = re.search(r'\bSL\b\s*[:=]\s*([0-9\.,]+)', remainder, flags=re.I)
        m_tp = re.search(r'\bTP\b\s*[:=]\s*([0-9\.,]+)', remainder, flags=re.I)
        if m_entry:
            entry = to_float_from_text(m_entry.group(1))
        if m_sl:
            sl = to_float_from_text(m_sl.group(1))
        if m_tp:
            tp = to_float_from_text(m_tp.group(1))
        # reason
        m_reason = re.search(r'REASON\s*[:=]\s*(.+?)(?:-?\s*CONF|$)', remainder, flags=re.I)
        if m_reason:
            reason = m_reason.group(1).strip()
        else:
            # fallback: try to capture text before CONF
            m_before_conf = re.search(r'(.+?)\s*CONF', remainder, flags=re.I)
            reason = m_before_conf.group(1).strip() if m_before_conf else remainder
        # conf
        m_conf = re.search(r'CONF(?:IDENCE)?\s*[:=]?\s*(\d{2,3})', remainder, flags=re.I)
        if m_conf:
            try:
                conf = int(m_conf.group(1))
            except:
                conf = None
        # final validation
        if action in ("BUY","SELL") and conf and conf >= 70:
            out[symbol] = {
                "action": action,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "reason": reason,
                "confidence": conf,
                "timestamp": datetime.now()
            }
    return out

# ---------------- Main Loop ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession() as session:
        startup_msg = f"""🤖 *GPT-driven Crypto Bot Online*
• Symbols: {len(SYMBOLS)} • Poll: {POLL_INTERVAL}s • Conf ≥ {SIGNAL_CONF_THRESHOLD}%"""
        await send_text(session, startup_msg)
        iteration = 0
        while True:
            try:
                iteration += 1
                print(f"\nIteration {iteration} @ {datetime.now()}")
                # fetch market data
                fetch_tasks = [fetch_enhanced_data(session, s) for s in SYMBOLS]
                results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                market = {}
                for symbol, result in zip(SYMBOLS, results):
                    if isinstance(result, Exception):
                        print(f"Fetch error {symbol}: {result}")
                        continue
                    if result and result.get("price") is not None:
                        market[symbol] = result
                if not market:
                    print("No market data -> sleeping")
                    await asyncio.sleep(min(60, POLL_INTERVAL))
                    continue
                # ask GPT for signals
                analysis_result = await enhanced_analyze_openai(market)
                if not analysis_result:
                    print("No analysis from AI -> sleeping")
                    await asyncio.sleep(POLL_INTERVAL)
                    continue
                signals = enhanced_parse(analysis_result)
                if signals:
                    print(f"Found {len(signals)} signals")
                else:
                    print("No high-confidence signals this iteration")
                # handle signals
                for symbol, sig in signals.items():
                    try:
                        confidence = sig["confidence"]
                        action = sig["action"]
                        entry = sig.get("entry")
                        sl = sig.get("sl")
                        tp = sig.get("tp")
                        reason = sig.get("reason", "")
                        # get market data for chart
                        symbol_data = market.get(symbol, {})
                        caption_price = entry if entry is not None else symbol_data.get("price")
                        # nice formatting
                        def fmt(p):
                            if p is None: return "N/A"
                            return f"{p:.6f}" if p < 1 else f"{p:.2f}"
                        caption = f"""🚨 *{symbol}* → *{action}*
💰 Entry: {fmt(caption_price)}
🛑 SL: {fmt(sl)}
🎯 TP: {fmt(tp)}
📊 Confidence: {confidence}%
🔎 Reason: {reason}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"""
                        # create & send chart if possible
                        if symbol_data.get("candles") and symbol_data.get("times"):
                            try:
                                chart_path = enhanced_plot_chart(symbol_data["times"], symbol_data["candles"], symbol, symbol_data)
                                await send_photo(session, caption, chart_path)
                            except Exception as e:
                                print(f"Chart error {symbol}: {e}")
                                await send_text(session, caption)
                        else:
                            await send_text(session, caption)
                        # track signal
                        signal_history.append({
                            "symbol": symbol, "action": action, "entry": entry, "sl": sl, "tp": tp,
                            "confidence": confidence, "timestamp": datetime.now(), "reason": reason
                        })
                        if len(signal_history) > 200:
                            signal_history.pop(0)
                    except Exception as e:
                        print(f"Error processing signal {symbol}: {e}")
                        traceback.print_exc()
                # periodic status
                if iteration % 10 == 0:
                    status = f"Status: iter={iteration}, scanned={len(market)}, signals_total={len(signal_history)}"
                    await send_text(session, status)
                print(f"Iteration {iteration} done → sleeping {POLL_INTERVAL}s")
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                print("Shutting down")
                raise
            except Exception as e:
                print("Main loop error:", e)
                traceback.print_exc()
                err_msg = f"⚠️ Bot error: {str(e)[:200]}"
                await send_text(session, err_msg)
                await asyncio.sleep(min(60, POLL_INTERVAL))

# ---------------- Run ----------------
if __name__ == "__main__":
    print("Starting GPT-driven Crypto Bot (30m TF, last up to 100 candles)...")
    try:
        asyncio.run(enhanced_loop())
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as e:
        print("Fatal:", e)
        traceback.print_exc()
