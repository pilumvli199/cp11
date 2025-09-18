# main.py - Enhanced Crypto Trading Bot v6.1 (with Entry / SL / TP / Position Size)
import os
import asyncio
import aiohttp
import time
import traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile
import numpy as np
import math
import shutil
import json
from typing import Dict, List, Optional, Tuple

load_dotenv()

# --- Enhanced Config ---
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

# Risk management / sizing defaults (can set via env)
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE") or 1000.0)  # default $1,000
RISK_PERCENTAGE = float(os.getenv("RISK_PERCENTAGE") or 1.0)   # % of account risked per trade
RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO") or 2.0)  # default 2:1

# New enhanced parameters
RSI_PERIOD = 14
MA_SHORT = 7
MA_LONG = 21
VOLUME_MULTIPLIER = 1.5
MIN_CANDLES_FOR_ANALYSIS = 50
LOOKBACK_PERIOD = 48

# Historical data tracking (in-memory for session)
price_history = {}
signal_history = []
success_rate_tracker = {}

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=100"
ORDER_BOOK_URL = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit=10"

# ---------------- Technical helpers ----------------
def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        ch = prices[i] - prices[i-1]
        gains.append(max(ch, 0))
        losses.append(max(-ch, 0))
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
    return volumes[-1] > avg_volume * VOLUME_MULTIPLIER

def enhanced_levels(candles, lookback=LOOKBACK_PERIOD):
    if not candles or len(candles) < 10:
        return (None, None, None, None, None)
    arr = candles[-lookback:] if len(candles) >= lookback else candles
    highs = [c[1] for c in arr]
    lows = [c[2] for c in arr]
    closes = [c[3] for c in arr]
    highs_sorted = sorted(highs, reverse=True)
    lows_sorted = sorted(lows)
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
    last_candle = candles[-1]
    body_size = abs(last_candle[3] - last_candle[0])
    range_size = last_candle[1] - last_candle[2]
    patterns['doji'] = (body_size / range_size) < 0.1 if range_size > 0 else False
    try:
        lower_wick = (last_candle[0] - last_candle[2]) if last_candle[0] > last_candle[2] else (last_candle[3] - last_candle[2])
        upper_wick = last_candle[1] - max(last_candle[0], last_candle[3])
    except Exception:
        lower_wick = upper_wick = 0
    patterns['hammer'] = (lower_wick > 2 * body_size) and (upper_wick < body_size) if body_size > 0 else False
    if len(candles) >= 2:
        prev, curr = candles[-2], candles[-1]
        patterns['bullish_engulfing'] = (curr[3] > curr[0]) and (prev[3] < prev[0]) and (curr[3] > prev[0]) and (curr[0] < prev[3])
        patterns['bearish_engulfing'] = (curr[3] < curr[0]) and (prev[3] > prev[0]) and (curr[3] < prev[0]) and (curr[0] > prev[3])
    return patterns

# ---------------- Data fetching ----------------
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
        price_history[symbol].append({
            "price": out["price"],
            "timestamp": datetime.now(),
            "volume": out.get("volume", 0)
        })
        if len(price_history[symbol]) > 100:
            price_history[symbol] = price_history[symbol][-100:]
    return out

# ---------------- OpenAI analysis ----------------
async def enhanced_analyze_openai(market):
    if not client:
        print("No OpenAI client configured.")
        return None
    analysis_parts = []
    for symbol, data in market.items():
        if not data.get("price"):
            continue
        price = data["price"]
        volume = data.get("volume", 0)
        price_change = data.get("price_change_24h", 0)
        part = f"\n{symbol} Analysis:"
        part += f"\nPrice: ${price} (24h change: {price_change:+.2f}%)"
        part += f"\nVolume: {volume:,.0f}"
        if data.get("rsi"):
            rsi = data["rsi"]
            rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            part += f"\nRSI: {rsi} ({rsi_signal})"
        if data.get("ma_short") and data.get("ma_long"):
            ma_signal = "Bullish" if data["ma_short"] > data["ma_long"] else "Bearish"
            part += f"\nMA Cross: {ma_signal} (Short: {data['ma_short']:.6f}, Long: {data['ma_long']:.6f})"
        if data.get("volume_spike"):
            part += f"\nVolume: SPIKE DETECTED"
        if data.get("buy_pressure"):
            pressure = "Strong Buy" if data["buy_pressure"] > 1.2 else "Strong Sell" if data["buy_pressure"] < 0.8 else "Balanced"
            part += f"\nOrder Book: {pressure} (Ratio: {data['buy_pressure']:.2f})"
        patterns = data.get("patterns", {})
        active_patterns = [k for k, v in patterns.items() if v]
        if active_patterns:
            part += f"\nPatterns: {', '.join(active_patterns)}"
        if data.get("candles"):
            sup1, res1, sup2, res2, mid = enhanced_levels(data["candles"])
            if sup1 and res1:
                distance_to_support = ((price - sup1) / sup1) * 100
                distance_to_resistance = ((res1 - price) / price) * 100
                part += f"\nS/R: Support at {sup1:.6f} ({distance_to_support:+.1f}%), Resistance at {res1:.6f} ({distance_to_resistance:+.1f}%)"
        if data.get("candles") and len(data["candles"]) >= 5:
            recent = data["candles"][-5:]
            candle_str = ",".join([f"[O:{c[0]:.6f},H:{c[1]:.6f},L:{c[2]:.6f},C:{c[3]:.6f}]" for c in recent])
            part += f"\nRecent 5 candles: {candle_str}"
        analysis_parts.append(part)
    if not analysis_parts:
        print("No market data available for enhanced analysis.")
        return None
    prompt = f"""You are an expert crypto trading analyst with deep knowledge of technical analysis, market psychology, and risk management.

ANALYSIS CONTEXT:
- Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
- Market conditions: Analyze for potential reversals, breakouts, and trend continuations
- Risk level: Only suggest trades with high probability setups

TECHNICAL ANALYSIS FOCUS:
1. Multi-timeframe confluence (look for alignment of indicators)
2. Volume confirmation (stronger signals when volume supports price action)
3. Support/resistance interactions (breakouts, rejections, retests)
4. Pattern recognition (engulfing, doji, hammer, etc.)
5. RSI divergences and extreme levels
6. Moving average trends and crossovers
7. Order book pressure analysis

SIGNAL CRITERIA (Only output if ALL conditions met):
- Minimum 70% confidence required
- Clear technical setup with multiple confirmations
- Proper risk/reward ratio (minimum 1:2)
- Volume supporting the move
- No conflicting signals from other indicators

OUTPUT FORMAT (one line per strong signal only):
SYMBOL - ACTION - DETAILED_REASON - CONF: XX%

Where:
- ACTION: BUY/SELL only (no NEUTRAL unless no clear setup)
- DETAILED_REASON: Must include specific technical levels, patterns, and confirmations
- CONF: 70-100% (below 70% should not be reported)

MARKET DATA:
{"".join(analysis_parts)}

Remember: Quality over quantity. Only suggest trades with clear edge and multiple confirmations."""
    try:
        loop = asyncio.get_running_loop()
        def call_model():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.1
            )
        resp = await loop.run_in_executor(None, call_model)
        try:
            choice = resp.choices[0]
            content = choice.message.content if hasattr(choice, "message") else getattr(choice, "text", None)
            if content is None:
                content = str(resp)
            return content.strip()
        except Exception:
            return str(resp)
    except Exception as e:
        print("Enhanced OpenAI call failed:", e)
        traceback.print_exc()
        return None

# ---------------- Plotting (same as earlier) ----------------
def enhanced_plot_chart(times, candles, symbol, market_data):
    if not times or not candles or len(times) != len(candles) or len(candles) < 10:
        raise ValueError("Insufficient data for enhanced plotting")
    dates = [datetime.utcfromtimestamp(int(t)) for t in times]
    closes = [c[3] for c in candles]
    x = date2num(dates)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=100, gridspec_kw={'height_ratios': [3, 1]})
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
    sup1, res1, sup2, res2, mid = enhanced_levels(candles)
    if res1:
        ax1.axhline(res1, color="red", linestyle="--", alpha=0.7,
                   label=f"Primary Resistance: {res1:.6f}" if res1 < 1 else f"Primary Resistance: {res1:.2f}")
    if sup1:
        ax1.axhline(sup1, color="blue", linestyle="--", alpha=0.7,
                   label=f"Primary Support: {sup1:.6f}" if sup1 < 1 else f"Primary Support: {sup1:.2f}")
    if res2:
        ax1.axhline(res2, color="orange", linestyle=":", alpha=0.5, label=f"Secondary Resistance: {res2:.2f}")
    if sup2:
        ax1.axhline(sup2, color="cyan", linestyle=":", alpha=0.5, label=f"Secondary Support: {sup2:.2f}")
    if len(closes) >= MA_LONG:
        ma_short_values, ma_long_values = [], []
        for i in range(len(closes)):
            if i >= MA_SHORT - 1:
                ma_short_values.append(sum(closes[i-MA_SHORT+1:i+1]) / MA_SHORT)
            else:
                ma_short_values.append(None)
            if i >= MA_LONG - 1:
                ma_long_values.append(sum(closes[i-MA_LONG+1:i+1]) / MA_LONG)
            else:
                ma_long_values.append(None)
        valid_short = [(x[i], ma) for i, ma in enumerate(ma_short_values) if ma is not None]
        valid_long = [(x[i], ma) for i, ma in enumerate(ma_long_values) if ma is not None]
        if valid_short:
            x_short, y_short = zip(*valid_short)
            ax1.plot(x_short, y_short, color="purple", linewidth=1.5, alpha=0.8, label=f"MA{MA_SHORT}")
        if valid_long:
            x_long, y_long = zip(*valid_long)
            ax1.plot(x_long, y_long, color="brown", linewidth=1.5, alpha=0.8, label=f"MA{MA_LONG}")
    if market_data.get("candles") and len(market_data["candles"]) >= RSI_PERIOD + 5:
        rsi_values = []
        for i in range(RSI_PERIOD, len(closes)):
            rsi = calculate_rsi(closes[:i+1], RSI_PERIOD)
            rsi_values.append(rsi)
        if rsi_values:
            rsi_x = x[-len(rsi_values):]
            ax2.plot(rsi_x, rsi_values, color="blue", linewidth=1.5)
            ax2.axhline(70, color="red", linestyle="--", alpha=0.7, label="Overbought (70)")
            ax2.axhline(30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)")
            ax2.axhline(50, color="gray", linestyle=":", alpha=0.5)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("RSI", fontsize=10)
            ax2.legend(fontsize="small")
            ax2.grid(True, alpha=0.3)
    current_price = closes[-1]
    rsi_current = market_data.get("rsi", "N/A")
    volume_status = "VOLUME_SPIKE" if market_data.get("volume_spike") else ""
    price_str = f"{current_price:.6f}" if current_price < 1 else f"{current_price:.2f}"
    title = f"{symbol} - Price: {price_str} | RSI: {rsi_current} {volume_status}"
    ax1.set_title(title, fontsize=12, fontweight="bold")
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
        try:
            os.remove(path)
        except Exception:
            pass

# ---------------- Parsing & Main Loop ----------------
def enhanced_parse(text):
    out = {}
    if not text:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or not any(x in line.upper() for x in ["BUY", "SELL"]):
            continue
        parts = [p.strip() for p in line.split(" - ")]
        if len(parts) < 3:
            continue
        symbol = parts[0].upper()
        action = parts[1].upper()
        reason = parts[2]
        conf = None
        conf_text = line.upper()
        if "CONF" in conf_text:
            try:
                idx = conf_text.index("CONF")
                remaining = conf_text[idx:]
                digits = "".join(c for c in remaining if c.isdigit())
                if digits:
                    conf = int(digits)
            except:
                pass
        if action in ["BUY", "SELL"] and conf and conf >= 70:
            out[symbol] = {"action": action, "reason": reason, "confidence": conf, "timestamp": datetime.now()}
    return out

def calculate_position_size(account_balance: float, risk_percentage: float, entry_price: float, stop_loss: float) -> float:
    if not all([account_balance, risk_percentage, entry_price, stop_loss]) or entry_price == stop_loss:
        return 0.0
    risk_amount = account_balance * (risk_percentage / 100.0)
    price_difference = abs(entry_price - stop_loss)
    if price_difference == 0:
        return 0.0
    position_size = risk_amount / price_difference
    return round(position_size, 6)

def calculate_take_profit(entry_price: float, stop_loss: float, risk_reward_ratio: float = 2.0) -> Optional[float]:
    if entry_price is None or stop_loss is None:
        return None
    risk = abs(entry_price - stop_loss)
    reward = risk * risk_reward_ratio
    if entry_price > stop_loss:
        return entry_price + reward
    else:
        return entry_price - reward

async def enhanced_loop():
    async with aiohttp.ClientSession() as session:
        startup_msg = f"""🤖 *Enhanced Crypto Bot v6.1 Online*

📊 *Configuration:*
• Symbols: {len(SYMBOLS)}
• Confidence Threshold: ≥{SIGNAL_CONF_THRESHOLD}%
• Poll Interval: {POLL_INTERVAL}s
• Account Balance: {ACCOUNT_BALANCE}
• Risk per Trade: {RISK_PERCENTAGE}%
• R:R: {RISK_REWARD_RATIO}:1
"""
        await send_text(session, startup_msg)
        iteration = 0
        while True:
            try:
                iteration += 1
                print(f"\n🔄 Starting iteration {iteration} at {datetime.now()}")
                fetch_tasks = [fetch_enhanced_data(session, symbol) for symbol in SYMBOLS]
                results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                market = {}
                for symbol, result in zip(SYMBOLS, results):
                    if isinstance(result, Exception):
                        print(f"Error fetching {symbol}: {result}")
                        continue
                    if result and result.get("price") is not None:
                        market[symbol] = result
                if not market:
                    print("No valid market data received, skipping analysis")
                    await asyncio.sleep(min(60, POLL_INTERVAL))
                    continue
                print(f"📊 Analyzing {len(market)} symbols...")
                analysis_result = await enhanced_analyze_openai(market)
                if not analysis_result:
                    print("No analysis result from AI, continuing...")
                    await asyncio.sleep(POLL_INTERVAL)
                    continue
                signals = enhanced_parse(analysis_result)
                if signals:
                    print(f"🎯 Found {len(signals)} potential signals")
                    for symbol, signal in signals.items():
                        try:
                            confidence = signal["confidence"]
                            action = signal["action"]
                            if confidence >= SIGNAL_CONF_THRESHOLD:
                                print(f"📢 High-confidence signal: {symbol} {action} ({confidence}%)")
                                symbol_data = market.get(symbol, {})
                                price = symbol_data.get("price") if symbol_data else None
                                # Calculate S/R for potential SL/TP anchors
                                sup1, res1, sup2, res2, mid = (None, None, None, None, None)
                                if symbol_data.get("candles"):
                                    sup1, res1, sup2, res2, mid = enhanced_levels(symbol_data["candles"])
                                # Determine stop-loss
                                stop_loss = None
                                if action == "BUY":
                                    if sup1 and sup1 < price:
                                        stop_loss = sup1
                                    else:
                                        # conservative 1% fallback
                                        stop_loss = price * 0.99 if price else None
                                else:  # SELL
                                    if res1 and res1 > price:
                                        stop_loss = res1
                                    else:
                                        stop_loss = price * 1.01 if price else None
                                # Validate stop_loss sensible distance (avoid tiny SL)
                                if price and stop_loss:
                                    if abs(price - stop_loss) / price < 0.0001:
                                        # if too close, widen to 0.5%
                                        if action == "BUY":
                                            stop_loss = price * 0.995
                                        else:
                                            stop_loss = price * 1.005
                                # Calculate TP using risk-reward
                                take_profit = calculate_take_profit(price, stop_loss, RISK_REWARD_RATIO) if (price and stop_loss) else None
                                # Position sizing
                                position_size = calculate_position_size(ACCOUNT_BALANCE, RISK_PERCENTAGE, price, stop_loss) if (price and stop_loss) else 0.0
                                # Build caption + send chart (if available)
                                if symbol_data.get("candles") and symbol_data.get("times"):
                                    try:
                                        chart_path = enhanced_plot_chart(symbol_data["times"], symbol_data["candles"], symbol, symbol_data)
                                        rsi = symbol_data.get("rsi", "N/A")
                                        volume_spike = "🔥 VOLUME SPIKE" if symbol_data.get("volume_spike") else ""
                                        patterns = symbol_data.get("patterns", {})
                                        active_patterns = [k.replace("_", " ").title() for k, v in patterns.items() if v]
                                        pattern_text = f"\n🔍 Patterns: {', '.join(active_patterns)}" if active_patterns else ""
                                        if price is None:
                                            price_display = "N/A"
                                        elif price < 1:
                                            price_display = f"{price:.6f}"
                                        else:
                                            price_display = f"{price:.2f}"
                                        sl_display = "N/A" if not stop_loss else (f"{stop_loss:.6f}" if stop_loss < 1 else f"{stop_loss:.2f}")
                                        tp_display = "N/A" if not take_profit else (f"{take_profit:.6f}" if take_profit < 1 else f"{take_profit:.2f}")
                                        caption = f"""🚨 *{symbol}* → *{action}*
💰 Entry: ${price_display}
🛡️ SL: ${sl_display}
🎯 TP: ${tp_display}
📦 Size: {position_size} units
📊 RSI: {rsi}
🎯 Confidence: {confidence}%
{volume_spike}
{pattern_text}

💡 *Analysis:*
{signal['reason']}

⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
                                        await send_photo(session, caption, chart_path)
                                        signal_history.append({
                                            "symbol": symbol,
                                            "action": action,
                                            "confidence": confidence,
                                            "price": price,
                                            "entry": price,
                                            "stop_loss": stop_loss,
                                            "take_profit": take_profit,
                                            "position_size": position_size,
                                            "timestamp": datetime.now(),
                                            "reason": signal["reason"]
                                        })
                                        if len(signal_history) > 50:
                                            signal_history.pop(0)
                                    except Exception as e:
                                        print(f"Error creating chart for {symbol}: {e}")
                                        # send text fallback
                                        if price is None:
                                            price_display = "N/A"
                                        elif price < 1:
                                            price_display = f"{price:.6f}"
                                        else:
                                            price_display = f"{price:.2f}"
                                        sl_display = "N/A" if not stop_loss else (f"{stop_loss:.6f}" if stop_loss < 1 else f"{stop_loss:.2f}")
                                        tp_display = "N/A" if not take_profit else (f"{take_profit:.6f}" if take_profit < 1 else f"{take_profit:.2f}")
                                        simple_caption = f"🚨 {symbol} → {action}\nEntry: ${price_display}\nSL: ${sl_display}\nTP: ${tp_display}\nSize: {position_size}\nConf: {confidence}%\n{signal['reason']}"
                                        await send_text(session, simple_caption)
                                else:
                                    # No chart available, send text-only
                                    if price is None:
                                        price_display = "N/A"
                                    elif price < 1:
                                        price_display = f"{price:.6f}"
                                    else:
                                        price_display = f"{price:.2f}"
                                    sl_display = "N/A" if not stop_loss else (f"{stop_loss:.6f}" if stop_loss < 1 else f"{stop_loss:.2f}")
                                    tp_display = "N/A" if not take_profit else (f"{take_profit:.6f}" if take_profit < 1 else f"{take_profit:.2f}")
                                    simple_caption = f"🚨 {symbol} → {action}\nEntry: ${price_display}\nSL: ${sl_display}\nTP: ${tp_display}\nSize: {position_size}\nConf: {confidence}%\n{signal['reason']}"
                                    await send_text(session, simple_caption)
                        except Exception as e:
                            print(f"Error processing signal for {symbol}: {e}")
                            traceback.print_exc()
                else:
                    print("No high-confidence signals found in this iteration")
                if iteration % 10 == 0:
                    total_signals = len(signal_history)
                    recent_signals = len([s for s in signal_history if (datetime.now() - s["timestamp"]).total_seconds() < 3600])
                    status_msg = f"""📈 *Bot Status Update - Iteration {iteration}*

🔍 *Market Scan:* {len(market)} symbols analyzed
📊 *Total Signals Today:* {total_signals}
🕐 *Signals Last Hour:* {recent_signals}
⚡ *Next Scan:* {POLL_INTERVAL}s

🤖 *Bot running smoothly...*"""
                    await send_text(session, status_msg)
                print(f"✅ Iteration {iteration} completed. Waiting {POLL_INTERVAL}s...")
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                print("Bot shutting down...")
                raise
            except Exception as e:
                print(f"Main loop error in iteration {iteration}: {e}")
                traceback.print_exc()
                error_msg = f"⚠️ *Bot Error - Iteration {iteration}*\n```\n{str(e)[:200]}\n```\nRetrying in 60s..."
                await send_text(session, error_msg)
                await asyncio.sleep(min(60, POLL_INTERVAL))

# ---------------- Performance tracking helpers (unchanged) ----------------
def calculate_signal_performance() -> dict:
    if len(signal_history) < 5:
        return {"insufficient_data": True}
    total_signals = len(signal_history)
    buy_signals = len([s for s in signal_history if s["action"] == "BUY"])
    sell_signals = len([s for s in signal_history if s["action"] == "SELL"])
    avg_confidence = sum(s["confidence"] for s in signal_history) / total_signals
    symbol_counts = {}
    for signal in signal_history:
        symbol = signal["symbol"]
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    most_active = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    recent_cutoff = datetime.now() - timedelta(hours=24)
    recent_signals = [s for s in signal_history if s["timestamp"] > recent_cutoff]
    return {
        "total_signals": total_signals,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "avg_confidence": round(avg_confidence, 1),
        "most_active_symbols": most_active,
        "signals_24h": len(recent_signals),
        "last_signal_time": signal_history[-1]["timestamp"] if signal_history else None
    }

# ---------------- Main ----------------
if __name__ == "__main__":
    print("🚀 Starting Enhanced Crypto Trading Bot v6.1...")
    print(f"📊 Monitoring {len(SYMBOLS)} symbols")
    print(f"🎯 Signal confidence threshold: {SIGNAL_CONF_THRESHOLD}%")
    print(f"⏱️  Poll interval: {POLL_INTERVAL} seconds")
    print(f"🔧 Enhanced features: RSI, MA, S/R, Patterns, Order Book")
    print(f"💼 Account Balance: {ACCOUNT_BALANCE}, Risk per trade: {RISK_PERCENTAGE}%")
    try:
        asyncio.run(enhanced_loop())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        traceback.print_exc()
