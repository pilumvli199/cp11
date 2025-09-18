# main.py - Price Action Focused Crypto Bot with Reduced Indicators
import os
import re
import asyncio
import aiohttp
import traceback
import numpy as np
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0))

# Simplified Analysis - Focus on Price Action
RSI_PERIOD = 14
MA_SHORT = 7
MA_MEDIUM = 21
MA_LONG = 50
VOLUME_MULTIPLIER = 2.0  # Higher threshold for volume spikes
MIN_CANDLES_FOR_ANALYSIS = 30
LOOKBACK_PERIOD = 100
FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

# Market condition tracking
price_history: Dict[str, List[Dict]] = {}
signal_history: List[Dict] = []
market_sentiment: Dict[str, str] = {}  # bullish/bearish/neutral
last_signals: Dict[str, datetime] = {}  # prevent spam signals
performance_tracking: List[Dict] = []

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=100"
ORDER_BOOK_URL = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"

# ---------------- Utility formatters ----------------
def fmt_price(p: Optional[float]) -> str:
    """Format numeric price: 6 decimals for sub-1 prices, 2 decimals otherwise."""
    if p is None:
        return "N/A"
    try:
        v = float(p)
    except Exception:
        return str(p)
    return f"{v:.6f}" if abs(v) < 1 else f"{v:.2f}"

def fmt_decimal(val, small_prec=6, large_prec=2) -> str:
    """Format a decimal with different precision depending on magnitude."""
    if val is None:
        return "N/A"
    try:
        v = float(val)
    except Exception:
        return str(val)
    return f"{v:.{small_prec}f}" if abs(v) < 1 else f"{v:.{large_prec}f}"

# ---------------- Simplified Indicators ----------------
def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Simplified RSI calculation"""
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

def calculate_market_structure(candles: List[List[float]]) -> Dict[str, any]:
    """Analyze market structure - Higher Highs, Lower Lows etc. (PRICE ACTION FOCUS)"""
    if len(candles) < 10:
        return {}
    
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    closes = [c[3] for c in candles]
    
    # Find peaks and troughs
    peaks = []
    troughs = []
    
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            peaks.append((i, highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            troughs.append((i, lows[i]))
    
    structure = {"trend": "sideways", "strength": 0, "swings": []}
    
    # Analyze swing points for price action
    if len(peaks) >= 2 and len(troughs) >= 2:
        # Check for higher highs and higher lows (uptrend)
        if peaks[-1][1] > peaks[-2][1] and troughs[-1][1] > troughs[-2][1]:
            structure["trend"] = "uptrend"
            structure["strength"] = 2
        # Check for lower highs and lower lows (downtrend)
        elif peaks[-1][1] < peaks[-2][1] and troughs[-1][1] < troughs[-2][1]:
            structure["trend"] = "downtrend"
            structure["strength"] = 2
    
    # Recent price action analysis
    recent_closes = closes[-5:]
    structure["recent_momentum"] = "neutral"
    if len(recent_closes) >= 3:
        if recent_closes[-1] > recent_closes[-2] > recent_closes[-3]:
            structure["recent_momentum"] = "bullish"
        elif recent_closes[-1] < recent_closes[-2] < recent_closes[-3]:
            structure["recent_momentum"] = "bearish"
    
    return structure

def enhanced_volume_analysis(volumes: List[float], prices: List[float]) -> Dict[str, any]:
    """Enhanced volume analysis with focus on price-volume relationship"""
    if len(volumes) < 20:
        return {}
    
    recent_volumes = volumes[-20:]
    avg_volume = np.mean(recent_volumes[:-1])
    current_volume = volumes[-1]
    
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
    
    # Price-Volume relationship (PRICE ACTION FOCUS)
    price_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
    
    analysis = {
        "volume_spike": volume_ratio > VOLUME_MULTIPLIER,
        "volume_ratio": round(volume_ratio, 2),
        "price_volume_confirmation": False,
        "volume_trend": "neutral"
    }
    
    # Volume trend analysis
    if len(volumes) >= 5:
        volume_ma_short = np.mean(volumes[-5:])
        volume_ma_long = np.mean(volumes[-20:])
        if volume_ma_short > volume_ma_long * 1.2:
            analysis["volume_trend"] = "increasing"
        elif volume_ma_short < volume_ma_long * 0.8:
            analysis["volume_trend"] = "decreasing"
    
    # Volume confirms price movement (KEY PRICE ACTION CONCEPT)
    if abs(price_change) > 0.008:  # 0.8% price change
        if (price_change > 0 and volume_ratio > 1.5) or (price_change < 0 and volume_ratio > 1.5):
            analysis["price_volume_confirmation"] = True
    
    return analysis

def detect_price_action_patterns(candles: List[List[float]]) -> Dict[str, bool]:
    """Detect price action patterns (simplified and focused)"""
    if len(candles) < 5:
        return {}
    
    patterns = {}
    recent = candles[-5:]
    
    # Support and Resistance breaks
    highs = [c[1] for c in recent]
    lows = [c[2] for c in recent]
    closes = [c[3] for c in recent]
    opens = [c[0] for c in recent]
    
    # Key resistance/support breaks
    if len(candles) > 20:
        prev_high = max([c[1] for c in candles[-21:-1]])
        prev_low = min([c[2] for c in candles[-21:-1]])
        
        if closes[-1] > prev_high:
            patterns['resistance_break'] = True
        if closes[-1] < prev_low:
            patterns['support_break'] = True
    
    # Simple candlestick patterns
    for i, candle in enumerate(recent):
        open_price, high, low, close = candle
        body = abs(close - open_price)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        range_size = high - low
        
        if range_size == 0:
            continue
            
        # Focus on high-probability patterns only
        if i == len(recent) - 1:  # Current candle
            # Engulfing patterns
            if i > 0:
                prev_open, prev_high, prev_low, prev_close = recent[i-1]
                prev_body = abs(prev_close - prev_open)
                
                # Bullish engulfing
                if (close > open_price and prev_close < prev_open and 
                    close > prev_open and open_price < prev_close and
                    body > prev_body * 1.2):
                    patterns['bullish_engulfing'] = True
                
                # Bearish engulfing
                if (close < open_price and prev_close > prev_open and 
                    close < prev_open and open_price > prev_close and
                    body > prev_body * 1.2):
                    patterns['bearish_engulfing'] = True
            
            # Pin bars (simplified)
            if (upper_wick > body * 2 and lower_wick < body * 0.5 and 
                close < open_price):
                patterns['shooting_star'] = True
            
            if (lower_wick > body * 2 and upper_wick < body * 0.5 and 
                close > open_price):
                patterns['hammer'] = True
    
    # Multi-candle patterns
    if len(recent) >= 3:
        # Inside bars (consolidation)
        if (highs[-1] <= highs[-2] and lows[-1] >= lows[-2] and
            highs[-2] <= highs[-3] and lows[-2] >= lows[-3]):
            patterns['double_inside_bar'] = True
        
        # Outside bars (breakout potential)
        if (highs[-1] > highs[-2] and lows[-1] < lows[-2] and
            highs[-2] > highs[-3] and lows[-2] < lows[-3]):
            patterns['double_outside_bar'] = True
    
    return patterns

def calculate_support_resistance(candles: List[List[float]]) -> Dict[str, float]:
    """Calculate key support and resistance levels from price action"""
    if len(candles) < 20:
        return {}
    
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    closes = [c[3] for c in candles]
    
    # Recent swing points
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            swing_lows.append(lows[i])
    
    levels = {}
    
    if swing_highs:
        levels['resistance_1'] = max(swing_highs[-3:]) if len(swing_highs) >= 3 else swing_highs[-1]
        levels['resistance_2'] = max(swing_highs) if swing_highs else None
    
    if swing_lows:
        levels['support_1'] = min(swing_lows[-3:]) if len(swing_lows) >= 3 else swing_lows[-1]
        levels['support_2'] = min(swing_lows) if swing_lows else None
    
    # Current price relative to levels
    current_price = closes[-1]
    if levels:
        if 'resistance_1' in levels and levels['resistance_1']:
            levels['distance_to_resistance'] = ((levels['resistance_1'] - current_price) / current_price) * 100
        if 'support_1' in levels and levels['support_1']:
            levels['distance_to_support'] = ((current_price - levels['support_1']) / current_price) * 100
    
    return levels

# ---------------- Enhanced Data Fetching ----------------
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
            print(f"Error processing ticker for {symbol}: {e}")
            out["price"] = None
    
    if isinstance(candles, list) and len(candles) >= MIN_CANDLES_FOR_ANALYSIS:
        try:
            # Parse candle data
            parsed_candles = []
            times = []
            volumes = []
            
            for x in candles:
                open_price = float(x[1])
                high = float(x[2])
                low = float(x[3])
                close = float(x[4])
                volume = float(x[5])
                timestamp = int(x[0]) // 1000
                
                parsed_candles.append([open_price, high, low, close])
                times.append(timestamp)
                volumes.append(volume)
            
            out["candles"] = parsed_candles
            out["times"] = times
            out["volumes"] = volumes
            
            # Calculate simplified indicators (PRICE ACTION FOCUS)
            closes = [c[3] for c in parsed_candles]
            highs = [c[1] for c in parsed_candles]
            lows = [c[2] for c in parsed_candles]
            
            # Basic indicators only
            out["rsi"] = calculate_rsi(closes, RSI_PERIOD)
            out["ma_short"] = sum(closes[-MA_SHORT:]) / MA_SHORT if len(closes) >= MA_SHORT else None
            out["ma_medium"] = sum(closes[-MA_MEDIUM:]) / MA_MEDIUM if len(closes) >= MA_MEDIUM else None
            out["ma_long"] = sum(closes[-MA_LONG:]) / MA_LONG if len(closes) >= MA_LONG else None
            
            # Price action analysis (FOCUS AREA)
            out["market_structure"] = calculate_market_structure(parsed_candles)
            out["volume_analysis"] = enhanced_volume_analysis(volumes, closes)
            out["patterns"] = detect_price_action_patterns(parsed_candles)
            out["key_levels"] = calculate_support_resistance(parsed_candles)
                
        except Exception as e:
            print(f"Enhanced candle processing error for {symbol}: {e}")
            traceback.print_exc()
    
    # Order book analysis for key levels
    if orderbook:
        try:
            bids = [(float(x[0]), float(x[1])) for x in orderbook.get("bids", [])]
            asks = [(float(x[0]), float(x[1])) for x in orderbook.get("asks", [])]
            
            if bids and asks:
                out["bid"] = bids[0][0]
                out["ask"] = asks[0][0]
                out["spread"] = asks[0][0] - bids[0][0]
                out["spread_pct"] = (out["spread"] / bids[0][0]) * 100
                
                # Significant order clusters
                avg_bid_size = np.mean([x[1] for x in bids[:5]])
                avg_ask_size = np.mean([x[1] for x in asks[:5]])
                
                significant_bids = [x for x in bids if x[1] > avg_bid_size * 2]
                significant_asks = [x for x in asks if x[1] > avg_ask_size * 2]
                
                out["ob_support"] = significant_bids[0][0] if significant_bids else None
                out["ob_resistance"] = significant_asks[0][0] if significant_asks else None
                
        except Exception as e:
            print(f"Order book processing error for {symbol}: {e}")
    
    # Update price history
    if symbol not in price_history:
        price_history[symbol] = []
    
    if out.get("price") is not None:
        price_history[symbol].append({
            "price": out["price"],
            "timestamp": datetime.now(),
            "volume": out.get("volume", 0),
            "rsi": out.get("rsi")
        })
        
        if len(price_history[symbol]) > 200:
            price_history[symbol] = price_history[symbol][-200:]
    
    return out

async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=20) as r:
            if r.status != 200:
                txt = await r.text() if r is not None else "<no body>"
                print(f"fetch_json {url} returned {r.status}: {txt[:200]}")
                return None
            return await r.json()
    except Exception as e:
        print("fetch_json exception for", url, e)
        return None

# ---------------- Enhanced AI Analysis with Price Action Focus ----------------
async def enhanced_analyze_openai(market):
    if not client:
        print("No OpenAI client configured.")
        return None
    
    # Prepare comprehensive market analysis with price action focus
    market_summary = []
    price_action_signals = []
    
    for symbol, data in market.items():
        if not data.get("price"):
            continue
        
        # Basic data
        price = data["price"]
        rsi = data.get("rsi")
        volume_analysis = data.get("volume_analysis", {})
        market_structure = data.get("market_structure", {})
        patterns = data.get("patterns", {})
        key_levels = data.get("key_levels", {})
        
        # Price action signal strength calculation
        signal_strength = 0
        signal_direction = "neutral"
        reasons = []
        
        # Market structure signals
        trend = market_structure.get("trend", "sideways")
        momentum = market_structure.get("recent_momentum", "neutral")
        
        if trend != "sideways":
            signal_strength += 1
            reasons.append(f"Trend: {trend}")
        
        if momentum != "neutral":
            signal_strength += 1
            signal_direction = momentum
            reasons.append(f"Momentum: {momentum}")
        
        # Key level breaks (STRONG PRICE ACTION SIGNAL)
        if patterns.get('resistance_break'):
            signal_strength += 3
            signal_direction = "bullish"
            reasons.append("Resistance break")
        elif patterns.get('support_break'):
            signal_strength += 3
            signal_direction = "bearish"
            reasons.append("Support break")
        
        # Candlestick patterns
        if patterns.get('bullish_engulfing'):
            signal_strength += 2
            signal_direction = "bullish"
            reasons.append("Bullish engulfing")
        elif patterns.get('bearish_engulfing'):
            signal_strength += 2
            signal_direction = "bearish"
            reasons.append("Bearish engulfing")
        
        if patterns.get('hammer'):
            signal_strength += 1
            if signal_direction != "bearish":
                signal_direction = "bullish"
            reasons.append("Hammer pattern")
        elif patterns.get('shooting_star'):
            signal_strength += 1
            if signal_direction != "bullish":
                signal_direction = "bearish"
            reasons.append("Shooting star")
        
        # Volume confirmation
        if volume_analysis.get("price_volume_confirmation"):
            signal_strength += 2
            reasons.append("Volume confirms price move")
        
        # RSI for confluence only
        if rsi:
            if rsi < 35 and signal_direction == "bullish":
                signal_strength += 1
                reasons.append("RSI oversold")
            elif rsi > 65 and signal_direction == "bearish":
                signal_strength += 1
                reasons.append("RSI overbought")
        
        # Create summary with price action focus
        summary = f"""
{symbol}: Price=${fmt_price(price)}, RSI={rsi}, Change24h={data.get('price_change_24h', 0):.2f}%
- Trend: {trend}, Momentum: {momentum}
- Volume: {volume_analysis.get('volume_ratio', 0):.2f}x avg, Spike={volume_analysis.get('volume_spike', False)}
- Patterns: {list(patterns.keys())}
- Key Levels: S1={fmt_price(key_levels.get('support_1'))}, R1={fmt_price(key_levels.get('resistance_1'))}
- Signal Strength: {signal_strength}/10, Direction: {signal_direction}"""
        
        market_summary.append(summary)
        
        # Only include symbols with strong price action signals
        if signal_strength >= 4:
            price_action_signals.append({
                "symbol": symbol,
                "strength": signal_strength,
                "direction": signal_direction,
                "reasons": reasons,
                "key_levels": key_levels
            })
    
    if not market_summary:
        print("No market data available for analysis.")
        return None
    
    # Enhanced prompt with price action focus
    prompt = f"""You are a professional price action trader with 10+ years experience. Analyze the provided 30-minute timeframe data and identify ONLY the highest probability trades based on PRICE ACTION.

STRICT REQUIREMENTS:
1. Only suggest trades with confidence ≥ 80%
2. Each signal MUST have specific ENTRY, STOPLOSS, and TARGET prices
3. Risk:Reward ratio must be at least 1:2
4. Focus on price action: breakouts, patterns, volume confirmation
5. Account for key support/resistance levels

OUTPUT FORMAT (one line per signal):
SYMBOL - ACTION - ENTRY: <exact_price> - SL: <exact_price> - TP: <exact_price> - REASON: <max_50_words> - CONF: <80-95>%

PRICE ACTION RULES:
- BUY signals: Resistance breaks + bullish patterns + volume confirmation + trend alignment
- SELL signals: Support breaks + bearish patterns + volume confirmation + trend alignment
- ENTRY: At breakout confirmation or pattern completion
- STOPLOSS: Beyond recent swing point or pattern invalidation level
- TARGET: Based on measured moves or next key level
- No signals if price action is unclear or conflicting

MARKET DATA:
{"".join(market_summary)}

PRICE ACTION SIGNALS (Strong setups):
{json.dumps([{"symbol": s["symbol"], "strength": s["strength"], "direction": s["direction"], "reasons": s["reasons"], "levels": s["key_levels"]} for s in price_action_signals], indent=2)}

Remember: Quality over quantity. Only suggest trades with clear price action confirmation."""

    try:
        loop = asyncio.get_running_loop()
        def call_model():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional price action trader specializing in cryptocurrency markets. You provide only high-probability trading signals based on clear price action patterns, breakouts, and volume confirmation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )
        
        resp = await loop.run_in_executor(None, call_model)
        
        try:
            content = resp.choices[0].message.content
            print("Enhanced OpenAI response:\n", content[:3000])
            return content.strip()
        except Exception:
            return str(resp)
            
    except Exception as e:
        print("Enhanced OpenAI call failed:", e)
        traceback.print_exc()
        return None

# ---------------- Signal Processing with Enhanced Validation ----------------
def enhanced_parse(text):
    out = {}
    if not text:
        return out
    
    for line in text.splitlines():
        line = line.strip()
        if not line or not any(k in line.upper() for k in ("BUY", "SELL")):
            continue
        
        parts = [p.strip() for p in line.split(" - ")]
        if len(parts) < 3:
            continue
        
        symbol = parts[0].upper().replace(" ", "")
        action = parts[1].upper()
        
        if action not in ["BUY", "SELL"]:
            continue
        
        entry = sl = tp = None
        reason = ""
        conf = None
        
        remainder = " - ".join(parts[2:])
        
        # Extract values with improved regex
        m_entry = re.search(r'ENTRY\s*[:=]\s*([0-9\.]+)', remainder, flags=re.I)
        m_sl = re.search(r'\bSL\b\s*[:=]\s*([0-9\.]+)', remainder, flags=re.I)
        m_tp = re.search(r'\bTP\b\s*[:=]\s*([0-9\.]+)', remainder, flags=re.I)
        m_conf = re.search(r'CONF(?:IDENCE)?\s*[:=]?\s*(\d{2,3})', remainder, flags=re.I)
        m_reason = re.search(r'REASON\s*[:=]\s*(.+?)(?:\s*-\s*CONF|$)', remainder, flags=re.I)
        
        if m_entry:
            entry = float(m_entry.group(1))
        if m_sl:
            sl = float(m_sl.group(1))
        if m_tp:
            tp = float(m_tp.group(1))
        if m_conf:
            conf = int(m_conf.group(1))
        if m_reason:
            reason = m_reason.group(1).strip()
        
        # Enhanced validation
        if not all([entry, sl, tp, conf]):
            print(f"Incomplete signal for {symbol}: entry={entry}, sl={sl}, tp={tp}, conf={conf}")
            continue
        
        if conf < SIGNAL_CONF_THRESHOLD:
            print(f"Low confidence signal for {symbol}: {conf}% < {SIGNAL_CONF_THRESHOLD}%")
            continue
        
        # Risk:Reward validation
        if action == "BUY":
            risk = entry - sl
            reward = tp - entry
        else:  # SELL
            risk = sl - entry
            reward = entry - tp
        
        if risk <= 0 or reward <= 0:
            print(f"Invalid risk/reward for {symbol}: risk={risk}, reward={reward}")
            continue
        
        risk_reward_ratio = reward / risk
        if risk_reward_ratio < 1.5:  # Minimum 1.5:1 R:R
            print(f"Poor risk:reward ratio for {symbol}: {risk_reward_ratio:.2f}")
            continue
        
        # Check for recent signals to avoid spam
        if symbol in last_signals:
            time_since_last = datetime.now() - last_signals[symbol]
            if time_since_last < timedelta(hours=2):
                print(f"Recent signal exists for {symbol}, skipping")
                continue
        
        out[symbol] = {
            "action": action,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "reason": reason,
            "confidence": conf,
            "risk_reward": round(risk_reward_ratio, 2),
            "timestamp": datetime.now()
        }
        
        # Update last signal time
        last_signals[symbol] = datetime.now()
    
    return out

# ---------------- Simplified Charting ----------------
def enhanced_plot_chart(times, candles, symbol, market_data):
    if not times or not candles or len(times) != len(candles) or len(candles) < 10:
        raise ValueError("Insufficient data for enhanced plotting")
    
    dates = [datetime.utcfromtimestamp(int(t)) for t in times]
    closes = [c[3] for c in candles]
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    x = date2num(dates)
    
    # Create simplified subplots
    fig = plt.figure(figsize=(16, 10), dpi=120)
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    ax_price = fig.add_subplot(gs[0])
    ax_volume = fig.add_subplot(gs[1])
    ax_rsi = fig.add_subplot(gs[2])
    
    # Enhanced candlestick plotting
    width = 0.6 * (x[1] - x[0]) if len(x) > 1 else 0.4
    
    for xi, candle in zip(x, candles):
        o, h, l, c = candle
        color = "#00ff00" if c >= o else "#ff0000"
        edge_color = "#008000" if c >= o else "#800000"
        
        # Candlestick body and wicks
        ax_price.vlines(xi, l, h, color=edge_color, linewidth=1.2, alpha=0.8)
        rect_height = abs(c - o) if abs(c - o) > 0.0001 else 0.0001
        rect = plt.Rectangle((xi - width/2, min(o, c)), width, rect_height,
                           facecolor=color, edgecolor=edge_color, alpha=0.9, linewidth=0.8)
        ax_price.add_patch(rect)
    
    # Support and Resistance levels from price action
    key_levels = market_data.get("key_levels", {})
    if key_levels.get('support_1'):
        ax_price.axhline(key_levels['support_1'], color="blue", linestyle="--", alpha=0.8, 
                        linewidth=2, label=f"Support: {fmt_price(key_levels['support_1'])}")
    if key_levels.get('resistance_1'):
        ax_price.axhline(key_levels['resistance_1'], color="red", linestyle="--", alpha=0.8, 
                        linewidth=2, label=f"Resistance: {fmt_price(key_levels['resistance_1'])}")
    
    # Moving Averages (simplified)
    if len(closes) >= MA_LONG:
        ma_medium_values, ma_long_values = [], []
        
        for i in range(len(closes)):
            if i >= MA_MEDIUM - 1:
                ma_medium_values.append(sum(closes[i-MA_MEDIUM+1:i+1]) / MA_MEDIUM)
            else:
                ma_medium_values.append(None)
                
            if i >= MA_LONG - 1:
                ma_long_values.append(sum(closes[i-MA_LONG+1:i+1]) / MA_LONG)
            else:
                ma_long_values.append(None)
        
        # Plot MAs
        valid_medium = [(x[i], ma) for i, ma in enumerate(ma_medium_values) if ma is not None]
        valid_long = [(x[i], ma) for i, ma in enumerate(ma_long_values) if ma is not None]
        
        if valid_medium:
            x_medium, y_medium = zip(*valid_medium)
            ax_price.plot(x_medium, y_medium, linewidth=2, alpha=0.8, label=f"MA{MA_MEDIUM}")
        if valid_long:
            x_long, y_long = zip(*valid_long)
            ax_price.plot(x_long, y_long, linewidth=2, alpha=0.8, label=f"MA{MA_LONG}")
    
    # Volume chart
    volumes = market_data.get("volumes", [])
    if volumes and len(volumes) == len(x):
        volume_colors = []
        for i in range(len(candles)):
            if i == 0:
                volume_colors.append("gray")
            elif candles[i][3] >= candles[i-1][3]:  # Close >= previous close
                volume_colors.append("green")
            else:
                volume_colors.append("red")
        
        ax_volume.bar(x, volumes, width=width, color=volume_colors, alpha=0.7)
        ax_volume.set_ylabel("Volume", fontsize=10)
        ax_volume.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # RSI chart
    if market_data.get("candles") and len(market_data["candles"]) >= RSI_PERIOD + 5:
        rsi_values = []
        for i in range(RSI_PERIOD, len(closes)):
            rsi = calculate_rsi(closes[:i+1], RSI_PERIOD)
            if rsi:
                rsi_values.append(rsi)
        
        if rsi_values:
            rsi_x = x[-len(rsi_values):]
            ax_rsi.plot(rsi_x, rsi_values, linewidth=2)
            ax_rsi.axhline(70, linestyle="--", alpha=0.7, linewidth=1, color="red")
            ax_rsi.axhline(30, linestyle="--", alpha=0.7, linewidth=1, color="green")
            ax_rsi.axhline(50, linestyle=":", alpha=0.5, linewidth=1, color="gray")
            ax_rsi.fill_between(rsi_x, 70, 100, alpha=0.2, color="red")
            ax_rsi.fill_between(rsi_x, 0, 30, alpha=0.2, color="green")
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_ylabel("RSI", fontsize=10)
            ax_rsi.grid(True, alpha=0.3)
    
    # Enhanced title with price action information
    current_price = closes[-1]
    rsi_current = market_data.get("rsi", "
