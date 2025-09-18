# main.py - Enhanced GPT-driven Crypto Bot with Improved Accuracy
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # Better model for analysis
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0))  # Higher threshold

# Enhanced Analysis windows
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_SHORT = 7
MA_MEDIUM = 21
MA_LONG = 50
BB_PERIOD = 20
BB_STD = 2
VOLUME_MULTIPLIER = 1.8  # More conservative
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

# ---------------- Enhanced Indicators ----------------
def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Improved RSI calculation with smoothing"""
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Use Wilder's smoothing method
    avg_gains = []
    avg_losses = []
    
    # First calculation
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    avg_gains.append(avg_gain)
    avg_losses.append(avg_loss)
    
    # Subsequent calculations with smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        avg_gains.append(avg_gain)
        avg_losses.append(avg_loss)
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def calculate_macd(prices: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate MACD, Signal line, and Histogram"""
    if len(prices) < MACD_SLOW:
        return None, None, None
    
    prices_array = np.array(prices)
    
    # Calculate EMAs
    def ema(data, span):
        alpha = 2 / (span + 1)
        ema_values = [data[0]]
        for price in data[1:]:
            ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
        return ema_values
    
    ema_fast = ema(prices, MACD_FAST)
    ema_slow = ema(prices, MACD_SLOW)
    
    macd_line = ema_fast[-1] - ema_slow[-1]
    
    # Calculate signal line (EMA of MACD)
    macd_values = [fast - slow for fast, slow in zip(ema_fast[MACD_SLOW-1:], ema_slow)]
    if len(macd_values) >= MACD_SIGNAL:
        signal_line = ema(macd_values, MACD_SIGNAL)[-1]
        histogram = macd_line - signal_line
        return round(macd_line, 6), round(signal_line, 6), round(histogram, 6)
    
    return round(macd_line, 6), None, None

def calculate_bollinger_bands(prices: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate Bollinger Bands"""
    if len(prices) < BB_PERIOD:
        return None, None, None
    
    recent_prices = prices[-BB_PERIOD:]
    sma = np.mean(recent_prices)
    std = np.std(recent_prices)
    
    upper_band = sma + (BB_STD * std)
    lower_band = sma - (BB_STD * std)
    
    return round(upper_band, 6), round(sma, 6), round(lower_band, 6)

def calculate_fibonacci_retracements(highs: List[float], lows: List[float]) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels"""
    if not highs or not lows:
        return {}
    
    recent_high = max(highs[-50:])  # Last 50 candles
    recent_low = min(lows[-50:])
    
    diff = recent_high - recent_low
    fib_levels = {}
    
    for level in FIBONACCI_LEVELS:
        fib_levels[f"fib_{level}"] = recent_low + (diff * level)
    
    return fib_levels

def detect_divergence(prices: List[float], rsi_values: List[float]) -> str:
    """Detect bullish/bearish divergence"""
    if len(prices) < 10 or len(rsi_values) < 10:
        return "none"
    
    recent_prices = prices[-10:]
    recent_rsi = rsi_values[-10:]
    
    # Simple divergence detection
    price_trend = "up" if recent_prices[-1] > recent_prices[0] else "down"
    rsi_trend = "up" if recent_rsi[-1] > recent_rsi[0] else "down"
    
    if price_trend == "down" and rsi_trend == "up":
        return "bullish_divergence"
    elif price_trend == "up" and rsi_trend == "down":
        return "bearish_divergence"
    
    return "none"

def calculate_market_structure(candles: List[List[float]]) -> Dict[str, any]:
    """Analyze market structure - Higher Highs, Lower Lows etc."""
    if len(candles) < 10:
        return {}
    
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    
    # Find peaks and troughs
    peaks = []
    troughs = []
    
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
            peaks.append((i, highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
            troughs.append((i, lows[i]))
    
    structure = {"trend": "sideways", "strength": 0}
    
    if len(peaks) >= 2 and len(troughs) >= 2:
        # Check for higher highs and higher lows (uptrend)
        if peaks[-1][1] > peaks[-2][1] and troughs[-1][1] > troughs[-2][1]:
            structure["trend"] = "uptrend"
            structure["strength"] = 2
        # Check for lower highs and lower lows (downtrend)
        elif peaks[-1][1] < peaks[-2][1] and troughs[-1][1] < troughs[-2][1]:
            structure["trend"] = "downtrend"
            structure["strength"] = 2
    
    return structure

def enhanced_volume_analysis(volumes: List[float], prices: List[float]) -> Dict[str, any]:
    """Enhanced volume analysis"""
    if len(volumes) < 20:
        return {}
    
    recent_volumes = volumes[-20:]
    avg_volume = np.mean(recent_volumes[:-1])
    current_volume = volumes[-1]
    
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
    
    # Price-Volume relationship
    price_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
    
    analysis = {
        "volume_spike": volume_ratio > VOLUME_MULTIPLIER,
        "volume_ratio": round(volume_ratio, 2),
        "price_volume_confirmation": False
    }
    
    # Volume confirms price movement
    if abs(price_change) > 0.01:  # 1% price change
        if (price_change > 0 and volume_ratio > 1.2) or (price_change < 0 and volume_ratio > 1.2):
            analysis["price_volume_confirmation"] = True
    
    return analysis

def detect_advanced_patterns(candles: List[List[float]]) -> Dict[str, bool]:
    """Detect advanced candlestick patterns"""
    if len(candles) < 5:
        return {}
    
    patterns = {}
    
    # Get recent candles
    recent = candles[-5:]
    
    for i, candle in enumerate(recent):
        open_price, high, low, close = candle
        body = abs(close - open_price)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        range_size = high - low
        
        if range_size == 0:
            continue
            
        # More sophisticated pattern detection
        if i == len(recent) - 1:  # Current candle
            # Shooting Star
            if (upper_wick > 2 * body and lower_wick < body * 0.1 and 
                close < open_price and i > 0 and recent[i-1][3] > recent[i-1][0]):
                patterns['shooting_star'] = True
            
            # Hammer
            if (lower_wick > 2 * body and upper_wick < body * 0.1 and 
                i > 0 and recent[i-1][3] < recent[i-1][0]):
                patterns['hammer'] = True
            
            # Inverted Hammer
            if (upper_wick > 2 * body and lower_wick < body * 0.1 and 
                close > open_price and i > 0 and recent[i-1][3] < recent[i-1][0]):
                patterns['inverted_hammer'] = True
    
    # Multi-candle patterns
    if len(recent) >= 3:
        # Morning Star
        if (recent[-3][3] < recent[-3][0] and  # First candle bearish
            abs(recent[-2][3] - recent[-2][0]) < (recent[-3][1] - recent[-3][2]) * 0.3 and  # Middle doji/small body
            recent[-1][3] > recent[-1][0] and  # Third candle bullish
            recent[-1][3] > (recent[-3][0] + recent[-3][3]) / 2):  # Third candle closes above midpoint of first
            patterns['morning_star'] = True
        
        # Evening Star
        if (recent[-3][3] > recent[-3][0] and  # First candle bullish
            abs(recent[-2][3] - recent[-2][0]) < (recent[-3][1] - recent[-3][2]) * 0.3 and  # Middle doji/small body
            recent[-1][3] < recent[-1][0] and  # Third candle bearish
            recent[-1][3] < (recent[-3][0] + recent[-3][3]) / 2):  # Third candle closes below midpoint of first
            patterns['evening_star'] = True
    
    return patterns

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
            
            # Calculate all indicators
            closes = [c[3] for c in parsed_candles]
            highs = [c[1] for c in parsed_candles]
            lows = [c[2] for c in parsed_candles]
            
            # Basic indicators
            out["rsi"] = calculate_rsi(closes, RSI_PERIOD)
            out["ma_short"] = sum(closes[-MA_SHORT:]) / MA_SHORT if len(closes) >= MA_SHORT else None
            out["ma_medium"] = sum(closes[-MA_MEDIUM:]) / MA_MEDIUM if len(closes) >= MA_MEDIUM else None
            out["ma_long"] = sum(closes[-MA_LONG:]) / MA_LONG if len(closes) >= MA_LONG else None
            
            # Advanced indicators
            macd, signal, histogram = calculate_macd(closes)
            out["macd"] = macd
            out["macd_signal"] = signal
            out["macd_histogram"] = histogram
            
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes)
            out["bb_upper"] = bb_upper
            out["bb_middle"] = bb_middle
            out["bb_lower"] = bb_lower
            
            # Market structure and patterns
            out["market_structure"] = calculate_market_structure(parsed_candles)
            out["volume_analysis"] = enhanced_volume_analysis(volumes, closes)
            out["patterns"] = detect_advanced_patterns(parsed_candles)
            out["fibonacci"] = calculate_fibonacci_retracements(highs, lows)
            
            # Calculate RSI for divergence detection
            if len(closes) >= 20:
                rsi_values = []
                for i in range(RSI_PERIOD, len(closes)):
                    rsi = calculate_rsi(closes[:i+1], RSI_PERIOD)
                    if rsi:
                        rsi_values.append(rsi)
                
                if len(rsi_values) >= 10:
                    out["divergence"] = detect_divergence(closes[-len(rsi_values):], rsi_values)
                
        except Exception as e:
            print(f"Enhanced candle processing error for {symbol}: {e}")
            traceback.print_exc()
    
    # Enhanced order book analysis
    if orderbook:
        try:
            bids = [(float(x[0]), float(x[1])) for x in orderbook.get("bids", [])]
            asks = [(float(x[0]), float(x[1])) for x in orderbook.get("asks", [])]
            
            if bids and asks:
                out["bid"] = bids[0][0]
                out["ask"] = asks[0][0]
                out["spread"] = asks[0][0] - bids[0][0]
                out["spread_pct"] = (out["spread"] / bids[0][0]) * 100
                
                # Calculate order book imbalance
                total_bid_volume = sum(x[1] for x in bids[:10])
                total_ask_volume = sum(x[1] for x in asks[:10])
                out["order_imbalance"] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
                
                # Support and resistance from order book
                significant_bids = [x for x in bids if x[1] > np.mean([b[1] for b in bids]) * 1.5]
                significant_asks = [x for x in asks if x[1] > np.mean([a[1] for a in asks]) * 1.5]
                
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

# ---------------- Enhanced AI Analysis ----------------
async def enhanced_analyze_openai(market):
    if not client:
        print("No OpenAI client configured.")
        return None
    
    # Prepare comprehensive market analysis
    market_summary = []
    technical_signals = []
    
    for symbol, data in market.items():
        if not data.get("price"):
            continue
        
        # Basic data
        price = data["price"]
        rsi = data.get("rsi")
        volume_analysis = data.get("volume_analysis", {})
        market_structure = data.get("market_structure", {})
        patterns = data.get("patterns", {})
        
        # Technical signal strength calculation
        signal_strength = 0
        signal_direction = "neutral"
        reasons = []
        
        # RSI signals
        if rsi:
            if rsi < 30:
                signal_strength += 2
                signal_direction = "bullish"
                reasons.append(f"RSI oversold ({rsi})")
            elif rsi > 70:
                signal_strength += 2
                signal_direction = "bearish"
                reasons.append(f"RSI overbought ({rsi})")
        
        # MACD signals
        macd = data.get("macd")
        macd_signal = data.get("macd_signal")
        if macd and macd_signal:
            if macd > macd_signal and data.get("macd_histogram", 0) > 0:
                signal_strength += 1
                if signal_direction != "bearish":
                    signal_direction = "bullish"
                reasons.append("MACD bullish crossover")
            elif macd < macd_signal and data.get("macd_histogram", 0) < 0:
                signal_strength += 1
                if signal_direction != "bullish":
                    signal_direction = "bearish"
                reasons.append("MACD bearish crossover")
        
        # Bollinger Bands
        bb_upper = data.get("bb_upper")
        bb_lower = data.get("bb_lower")
        if bb_upper and bb_lower:
            if price <= bb_lower:
                signal_strength += 1
                reasons.append("Price at BB lower band")
            elif price >= bb_upper:
                signal_strength += 1
                reasons.append("Price at BB upper band")
        
        # Volume confirmation
        if volume_analysis.get("price_volume_confirmation"):
            signal_strength += 1
            reasons.append("Volume confirms price move")
        
        # Pattern recognition
        strong_patterns = ["morning_star", "evening_star", "hammer", "shooting_star"]
        for pattern in strong_patterns:
            if patterns.get(pattern):
                signal_strength += 2
                reasons.append(f"Strong pattern: {pattern}")
        
        # Market structure
        trend = market_structure.get("trend", "sideways")
        if trend != "sideways":
            signal_strength += 1
            reasons.append(f"Market structure: {trend}")
        
        # Create summary
        summary = f"""
{symbol}: Price=${price:.6f if price < 1 else price:.2f}, RSI={rsi}, Change24h={data.get('price_change_24h', 0):.2f}%
- MACD: {macd:.6f if macd and abs(macd) < 1 else macd}, Signal: {macd_signal}
- BB: Upper={bb_upper:.6f if bb_upper and bb_upper < 1 else bb_upper}, Lower={bb_lower:.6f if bb_lower and bb_lower < 1 else bb_lower}
- Volume: {volume_analysis.get('volume_ratio', 0):.2f}x avg, Spike={volume_analysis.get('volume_spike', False)}
- Structure: {trend}, Patterns: {list(patterns.keys())}
- Signal Strength: {signal_strength}/10, Direction: {signal_direction}
- Key Levels: Support={data.get('ob_support')}, Resistance={data.get('ob_resistance')}"""
        
        market_summary.append(summary)
        
        # Only include symbols with strong signals
        if signal_strength >= 4:
            technical_signals.append({
                "symbol": symbol,
                "strength": signal_strength,
                "direction": signal_direction,
                "reasons": reasons,
                "price": price,
                "data": data
            })
    
    if not market_summary:
        print("No market data available for analysis.")
        return None
    
    # Enhanced prompt with more specific instructions
    prompt = f"""You are a professional crypto trader with 10+ years experience. Analyze the provided 30-minute timeframe data and identify ONLY the highest probability trades.

STRICT REQUIREMENTS:
1. Only suggest trades with confidence ≥ 80%
2. Each signal MUST have specific ENTRY, STOPLOSS, and TARGET prices
3. Risk:Reward ratio must be at least 1:2
4. Consider market structure, volume, and multiple confirmations
5. Account for current market conditions and volatility

OUTPUT FORMAT (one line per signal):
SYMBOL - ACTION - ENTRY: <exact_price> - SL: <exact_price> - TP: <exact_price> - REASON: <max_50_words> - CONF: <80-95>%

ANALYSIS RULES:
- BUY signals: RSI<40 + bullish patterns + volume confirmation + support levels
- SELL signals: RSI>60 + bearish patterns + volume confirmation + resistance levels
- ENTRY: Near current price or breakout level
- STOPLOSS: Beyond recent swing high/low with 0.5-2% buffer
- TARGET: Based on key resistance/support levels with min 1:2 R:R
- No signals if trend is unclear or conflicting indicators

MARKET DATA:
{"".join(market_summary)}

PRIORITY SIGNALS (Strong technical setups):
{json.dumps([{"symbol": s["symbol"], "strength": s["strength"], "direction": s["direction"], "reasons": s["reasons"]} for s in technical_signals], indent=2)}

Remember: Quality over quantity. Only suggest trades you would take with your own money."""
    
    try:
        loop = asyncio.get_running_loop()
        def call_model():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional cryptocurrency trader with deep expertise in technical analysis. You provide only high-probability trading signals with precise entry, stop-loss, and take-profit levels."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1  # Lower temperature for more consistent analysis
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

# ---------------- Enhanced Charting ----------------
def enhanced_plot_chart(times, candles, symbol, market_data):
    if not times or not candles or len(times) != len(candles) or len(candles) < 10:
        raise ValueError("Insufficient data for enhanced plotting")
    
    dates = [datetime.utcfromtimestamp(int(t)) for t in times]
    closes = [c[3] for c in candles]
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    x = date2num(dates)
    
    # Create subplots with better layout
    fig = plt.figure(figsize=(16, 12), dpi=120)
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
    
    ax_price = fig.add_subplot(gs[0])
    ax_volume = fig.add_subplot(gs[1])
    ax_rsi = fig.add_subplot(gs[2])
    ax_macd = fig.add_subplot(gs[3])
    
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
    
    # Support and Resistance levels
    sup1, res1, sup2, res2, mid = enhanced_levels(candles, lookback=LOOKBACK_PERIOD)
    if res1:
        ax_price.axhline(res1, color="red", linestyle="--", alpha=0.8, linewidth=2, 
                        label=f"Primary Resistance: {res1:.6f if res1 < 1 else res1:.2f}")
    if sup1:
        ax_price.axhline(sup1, color="blue", linestyle="--", alpha=0.8, linewidth=2,
                        label=f"Primary Support: {sup1:.6f if sup1 < 1 else sup1:.2f}")
    if res2:
        ax_price.axhline(res2, color="orange", linestyle=":", alpha=0.6, linewidth=1.5,
                        label=f"Secondary Resistance: {res2:.2f}")
    if sup2:
        ax_price.axhline(sup2, color="cyan", linestyle=":", alpha=0.6, linewidth=1.5,
                        label=f"Secondary Support: {sup2:.2f}")
    
    # Bollinger Bands
    bb_upper = market_data.get("bb_upper")
    bb_middle = market_data.get("bb_middle")
    bb_lower = market_data.get("bb_lower")
    
    if bb_upper and bb_middle and bb_lower:
        bb_upper_line = [bb_upper] * len(x)
        bb_middle_line = [bb_middle] * len(x)
        bb_lower_line = [bb_lower] * len(x)
        
        ax_price.plot(x, bb_upper_line, color="purple", alpha=0.5, linewidth=1, label="BB Upper")
        ax_price.plot(x, bb_middle_line, color="gray", alpha=0.5, linewidth=1, label="BB Middle")
        ax_price.plot(x, bb_lower_line, color="purple", alpha=0.5, linewidth=1, label="BB Lower")
        ax_price.fill_between(x, bb_upper_line, bb_lower_line, alpha=0.1, color="purple")
    
    # Moving Averages
    if len(closes) >= MA_LONG:
        ma_short_values, ma_medium_values, ma_long_values = [], [], []
        
        for i in range(len(closes)):
            if i >= MA_SHORT - 1:
                ma_short_values.append(sum(closes[i-MA_SHORT+1:i+1]) / MA_SHORT)
            else:
                ma_short_values.append(None)
                
            if i >= MA_MEDIUM - 1:
                ma_medium_values.append(sum(closes[i-MA_MEDIUM+1:i+1]) / MA_MEDIUM)
            else:
                ma_medium_values.append(None)
                
            if i >= MA_LONG - 1:
                ma_long_values.append(sum(closes[i-MA_LONG+1:i+1]) / MA_LONG)
            else:
                ma_long_values.append(None)
        
        # Plot MAs
        valid_short = [(x[i], ma) for i, ma in enumerate(ma_short_values) if ma is not None]
        valid_medium = [(x[i], ma) for i, ma in enumerate(ma_medium_values) if ma is not None]
        valid_long = [(x[i], ma) for i, ma in enumerate(ma_long_values) if ma is not None]
        
        if valid_short:
            x_short, y_short = zip(*valid_short)
            ax_price.plot(x_short, y_short, color="yellow", linewidth=2, alpha=0.8, label=f"MA{MA_SHORT}")
        if valid_medium:
            x_medium, y_medium = zip(*valid_medium)
            ax_price.plot(x_medium, y_medium, color="orange", linewidth=2, alpha=0.8, label=f"MA{MA_MEDIUM}")
        if valid_long:
            x_long, y_long = zip(*valid_long)
            ax_price.plot(x_long, y_long, color="red", linewidth=2, alpha=0.8, label=f"MA{MA_LONG}")
    
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
            ax_rsi.plot(rsi_x, rsi_values, linewidth=2, color="blue")
            ax_rsi.axhline(70, color="red", linestyle="--", alpha=0.7, linewidth=1)
            ax_rsi.axhline(30, color="green", linestyle="--", alpha=0.7, linewidth=1)
            ax_rsi.axhline(50, color="gray", linestyle=":", alpha=0.5, linewidth=1)
            ax_rsi.fill_between(rsi_x, 70, 100, alpha=0.2, color="red")
            ax_rsi.fill_between(rsi_x, 0, 30, alpha=0.2, color="green")
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_ylabel("RSI", fontsize=10)
            ax_rsi.grid(True, alpha=0.3)
    
    # MACD chart
    macd = market_data.get("macd")
    macd_signal = market_data.get("macd_signal")
    macd_histogram = market_data.get("macd_histogram")
    
    if macd and macd_signal:
        # Simple representation for current values
        ax_macd.axhline(macd, color="blue", linewidth=2, label=f"MACD: {macd:.6f}")
        ax_macd.axhline(macd_signal, color="red", linewidth=2, label=f"Signal: {macd_signal:.6f}")
        if macd_histogram:
            ax_macd.axhline(0, color="gray", linestyle="-", alpha=0.5)
            ax_macd.bar([x[-1]], [macd_histogram], width=width*5, 
                       color="green" if macd_histogram > 0 else "red", alpha=0.7)
        ax_macd.set_ylabel("MACD", fontsize=10)
        ax_macd.legend(fontsize="small")
        ax_macd.grid(True, alpha=0.3)
    
    # Enhanced title with more information
    current_price = closes[-1]
    rsi_current = market_data.get("rsi", "N/A")
    price_change_24h = market_data.get("price_change_24h", 0)
    volume_ratio = market_data.get("volume_analysis", {}).get("volume_ratio", 0)
    trend = market_data.get("market_structure", {}).get("trend", "sideways")
    
    price_str = f"{current_price:.6f}" if current_price < 1 else f"{current_price:.2f}"
    
    title = f"""{symbol} | Price: ${price_str} | 24h: {price_change_24h:+.2f}% | RSI: {rsi_current}
Volume: {volume_ratio:.1f}x | Trend: {trend.upper()} | TF: 30m"""
    
    ax_price.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax_price.legend(loc="upper left", fontsize="small", framealpha=0.9)
    ax_price.grid(True, alpha=0.3)
    
    # Format dates
    fig.autofmt_xdate()
    
    # Save chart
    plt.tight_layout()
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight", dpi=120, facecolor='white')
    plt.close(fig)
    
    return tmp.name

def enhanced_levels(candles, lookback=LOOKBACK_PERIOD):
    """Enhanced support and resistance calculation"""
    if not candles or len(candles) < 10:
        return (None, None, None, None, None)
    
    arr = candles[-min(len(candles), lookback):]
    highs = [c[1] for c in arr]
    lows = [c[2] for c in arr]
    closes = [c[3] for c in arr]
    
    # Use weighted approach for better level detection
    recent_weight = 1.5
    older_weight = 1.0
    
    weighted_highs = []
    weighted_lows = []
    
    for i, (high, low) in enumerate(zip(highs, lows)):
        weight = recent_weight if i >= len(highs) * 0.7 else older_weight
        weighted_highs.extend([high] * int(weight * 10))
        weighted_lows.extend([low] * int(weight * 10))
    
    # Calculate support and resistance levels
    highs_sorted = sorted(weighted_highs, reverse=True)
    lows_sorted = sorted(weighted_lows)
    
    # Primary levels (strongest)
    primary_resistance = np.mean(highs_sorted[:30]) if len(highs_sorted) >= 30 else None
    primary_support = np.mean(lows_sorted[:30]) if len(lows_sorted) >= 30 else None
    
    # Secondary levels
    secondary_resistance = np.mean(highs_sorted[30:60]) if len(highs_sorted) >= 60 else None
    secondary_support = np.mean(lows_sorted[30:60]) if len(lows_sorted) >= 60 else None
    
    # Middle level
    current_price = closes[-1]
    if primary_resistance and primary_support:
        mid_level = (primary_resistance + primary_support) / 2
    else:
        mid_level = current_price
    
    return primary_support, primary_resistance, secondary_support, secondary_resistance, mid_level

# ---------------- Telegram Functions ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured. Skipping send_text.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    try:
        async with session.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": text, 
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }) as r:
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
            data.add_field("parse_mode", "Markdown")
            data.add_field("photo", f, filename=os.path.basename(path), content_type="image/png")
            
            async with session.post(url, data=data, timeout=90) as r:
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

# ---------------- Performance Tracking ----------------
def track_signal_performance():
    """Track and analyze signal performance"""
    if not performance_tracking:
        return
    
    total_signals = len(performance_tracking)
    profitable = sum(1 for p in performance_tracking if p.get("profit", 0) > 0)
    
    if total_signals > 0:
        win_rate = (profitable / total_signals) * 100
        avg_profit = np.mean([p.get("profit", 0) for p in performance_tracking])
        
        print(f"Performance: {total_signals} signals, {win_rate:.1f}% win rate, avg profit: {avg_profit:.2f}%")

# ---------------- Main Enhanced Loop ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        startup_msg = f"""🤖 *ENHANCED Crypto Trading Bot v2.0* 🚀

📊 *Configuration:*
• Symbols: {len(SYMBOLS)} pairs
• Timeframe: 30 minutes  
• Poll Interval: {POLL_INTERVAL}s
• Confidence Threshold: ≥{SIGNAL_CONF_THRESHOLD}%
• Min Risk:Reward: 1.5:1

🎯 *Features:*
• Advanced Technical Analysis
• MACD, RSI, Bollinger Bands
• Market Structure Analysis
• Volume Confirmation
• Pattern Recognition
• Performance Tracking

✅ Bot is now ONLINE and monitoring markets..."""
        
        await send_text(session, startup_msg)
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                start_time = datetime.now()
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration} @ {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                
                # Fetch market data for all symbols
                print("Fetching market data...")
                fetch_tasks = [fetch_enhanced_data(session, symbol) for symbol in SYMBOLS]
                results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                
                market = {}
                fetch_errors = 0
                
                for symbol, result in zip(SYMBOLS, results):
                    if isinstance(result, Exception):
                        print(f"❌ Fetch error {symbol}: {result}")
                        fetch_errors += 1
                        continue
                    
                    if result and result.get("price") is not None:
                        market[symbol] = result
                        print(f"✅ {symbol}: ${result['price']:.6f if result['price'] < 1 else result['price']:.2f}")
                    else:
                        print(f"⚠️  No price data for {symbol}")
                
                if not market:
                    print("❌ No market data available - sleeping...")
                    await asyncio.sleep(min(120, POLL_INTERVAL))
                    continue
                
                print(f"\n📊 Successfully fetched data for {len(market)}/{len(SYMBOLS)} symbols")
                if fetch_errors > 0:
                    print(f"⚠️  {fetch_errors} fetch errors")
                
                # Analyze with OpenAI
                print("\n🤖 Running AI analysis...")
                analysis_start = datetime.now()
                analysis_result = await enhanced_analyze_openai(market)
                analysis_time = (datetime.now() - analysis_start).total_seconds()
                
                if not analysis_result:
                    print("❌ No analysis result from AI - sleeping...")
                    await asyncio.sleep(POLL_INTERVAL)
                    continue
                
                print(f"✅ AI analysis completed in {analysis_time:.1f}s")
                
                # Parse signals
                signals = enhanced_parse(analysis_result)
                
                if signals:
                    print(f"\n🚨 Found {len(signals)} HIGH-CONFIDENCE signals:")
                    for symbol, sig in signals.items():
                        print(f"   {symbol}: {sig['action']} @ {sig['entry']:.6f if sig['entry'] < 1 else sig['entry']:.2f} | Conf: {sig['confidence']}% | R:R: {sig['risk_reward']}")
                else:
                    print("📈 No high-confidence signals this iteration")
                
                # Process and send signals
                signals_sent = 0
                for symbol, sig in signals.items():
                    try:
                        confidence = sig["confidence"]
                        action = sig["action"]
                        entry = sig["entry"]
                        sl = sig["sl"]
                        tp = sig["tp"]
                        reason = sig.get("reason", "")
                        risk_reward = sig["risk_reward"]
                        
                        symbol_data = market.get(symbol, {})
                        current_price = symbol_data.get("price", entry)
                        
                        # Enhanced formatting
                        def fmt_price(p):
                            if p is None: 
                                return "N/A"
                            return f"{p:.6f}" if p < 1 else f"{p:.2f}"
                        
                        # Calculate potential profit percentage
                        if action == "BUY":
                            potential_profit = ((tp - entry) / entry) * 100
                            risk_pct = ((entry - sl) / entry) * 100
                        else:
                            potential_profit = ((entry - tp) / entry) * 100
                            risk_pct = ((sl - entry) / entry) * 100
                        
                        # Enhanced signal message
                        signal_emoji = "🟢" if action == "BUY" else "🔴"
                        
                        caption = f"""{signal_emoji} *SIGNAL ALERT* {signal_emoji}

🎯 *{symbol}* → *{action}*
💰 Entry: `{fmt_price(entry)}`
🛑 Stop Loss: `{fmt_price(sl)}`
🎯 Take Profit: `{fmt_price(tp)}`

📊 *Analysis:*
• Confidence: *{confidence}%*
• Risk:Reward: *1:{risk_reward}*
• Risk: {risk_pct:.1f}%
• Potential: +{potential_profit:.1f}%

🔍 *Reason:* {reason}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
🤖 Enhanced Bot v2.0"""
                        
                        # Create and send chart
                        if symbol_data.get("candles") and symbol_data.get("times"):
                            try:
                                print(f"📈 Creating chart for {symbol}...")
                                chart_path = enhanced_plot_chart(
                                    symbol_data["times"], 
                                    symbol_data["candles"], 
                                    symbol, 
                                    symbol_data
                                )
                                await send_photo(session, caption, chart_path)
                                signals_sent += 1
                                print(f"✅ Signal sent for {symbol}")
                            except Exception as e:
                                print(f"❌ Chart error {symbol}: {e}")
                                await send_text(session, caption)
                                signals_sent += 1
                        else:
                            await send_text(session, caption)
                            signals_sent += 1
                        
                        # Store for performance tracking
                        performance_tracking.append({
                            "symbol": symbol,
                            "action": action,
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "confidence": confidence,
                            "risk_reward": risk_reward,
                            "timestamp": datetime.now(),
                            "reason": reason
                        })
                        
                        # Limit history size
                        if len(performance_tracking) > 500:
                            performance_tracking[:] = performance_tracking[-400:]
                        
                    except Exception as e:
                        print(f"❌ Error processing signal {symbol}: {e}")
                        traceback.print_exc()
                
                # Status updates
                iteration_time = (datetime.now() - start_time).total_seconds()
                
                if signals_sent > 0:
                    print(f"\n✅ Sent {signals_sent} signals in {iteration_time:.1f}s")
                
                # Periodic status and performance report
                if iteration % 12 == 0:  # Every ~6 hours if 30min intervals
                    track_signal_performance()
                    
                    status_msg = f"""📊 *Status Report* - Iteration {iteration}

🔍 *Market Scan:* {len(market)}/{len(SYMBOLS)} pairs
🚨 *Total Signals:* {len(performance_tracking)}
⏱️ *Uptime:* {iteration * POLL_INTERVAL // 3600:.1f} hours
🎯 *Last Analysis:* {analysis_time:.1f}s

✅ Bot running smoothly..."""
                    
                    await send_text(session, status_msg)
                
                print(f"\n⏰ Iteration {iteration} completed in {iteration_time:.1f}s")
                print(f"💤 Sleeping for {POLL_INTERVAL}s...")
                
                await asyncio.sleep(POLL_INTERVAL)
                
            except asyncio.CancelledError:
                print("\n🛑 Shutdown signal received")
                await send_text(session, "🤖 Bot shutting down gracefully...")
                break
                
            except Exception as e:
                print(f"\n❌ MAIN LOOP ERROR: {e}")
                traceback.print_exc()
                
                error_msg = f"""⚠️ *Bot Error Alert*

Error: `{str(e)[:150]}`
Time: {datetime.now().strftime('%H:%M:%S')}
Iteration: {iteration}

Bot will retry in {min(120, POLL_INTERVAL)}s..."""
                
                await send_text(session, error_msg)
                await asyncio.sleep(min(120, POLL_INTERVAL))

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    ENHANCED CRYPTO TRADING BOT v2.0                 ║
    ║                        with Advanced Analytics                       ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  Features: MACD, RSI, Bollinger Bands, Market Structure Analysis,   ║
    ║  Volume Confirmation, Pattern Recognition, Performance Tracking      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(enhanced_loop())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        traceback.print_exc()
