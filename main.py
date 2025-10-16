import os
import time
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import openai
import asyncio
from telegram import Bot
from telegram.error import TelegramError
import redis
import hashlib

# Load environment variables
load_dotenv()

# Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
REDIS_URL = os.getenv("REDIS_URL")
DERIBIT_API_URL = "https://www.deribit.com/api/v2/public"
SCAN_INTERVAL = 1800  # 30 minutes in seconds
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", 500))  # Max Redis memory for data

# DeepSeek Client Setup
deepseek_client = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# Telegram Bot Setup
telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Redis Connection
def get_redis_client():
    """Connect to Redis from URL or localhost"""
    try:
        if REDIS_URL:
            # Redis from Railway.app or other cloud service
            redis_client = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
        else:
            # Local Redis fallback
            redis_client = redis.Redis(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5
            )
        
        redis_client.ping()
        print("✅ Redis connected successfully")
        if REDIS_URL:
            print("📍 Using Railway.app Redis")
        else:
            print("📍 Using Local Redis")
        return redis_client
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("⚠️  Bot will continue without Redis caching")
        return None

redis_client = get_redis_client()


class CryptoAnalyzer:
    """Main crypto analysis bot with DeepSeek V3 + Redis caching"""
    
    def __init__(self):
        self.symbols = ["BTC", "ETH"]
        self.timeframe = "30"  # 30 minutes
        self.ema_periods = [9, 21, 50, 200]
        self.redis_client = redis_client
        self.option_chain_cache_key = "option_chain_data"
        self.signal_history_key = "signal_history"
        
    def _get_cache_key(self, symbol, data_type):
        """Generate cache key for Redis"""
        return f"crypto_analysis:{symbol}:{data_type}:{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    def _manage_redis_memory(self):
        """Delete old data if memory is getting full"""
        if not self.redis_client:
            return
        
        try:
            info = self.redis_client.info('memory')
            used_memory_mb = info['used_memory'] / (1024 * 1024)
            
            if used_memory_mb > MAX_MEMORY_MB:
                print(f"⚠️  Redis memory high ({used_memory_mb:.2f}MB). Cleaning old data...")
                
                # Get all keys
                all_keys = self.redis_client.keys("crypto_analysis:*")
                
                # Sort by timestamp and delete oldest 20%
                if all_keys:
                    keys_to_delete = sorted(all_keys)[:len(all_keys) // 5]
                    if keys_to_delete:
                        self.redis_client.delete(*keys_to_delete)
                        print(f"🗑️  Deleted {len(keys_to_delete)} old cache entries")
        except Exception as e:
            print(f"❌ Error managing Redis memory: {e}")
    
    def _store_option_chain(self, symbol, option_data):
        """Store option chain data in Redis for comparison"""
        if not self.redis_client or not option_data:
            return
        
        try:
            key = f"{self.option_chain_cache_key}:{symbol}"
            timestamp = datetime.now().isoformat()
            
            data_with_time = {
                "timestamp": timestamp,
                "data": option_data
            }
            
            # Store as JSON
            self.redis_client.setex(
                key,
                3600,  # 1 hour expiry
                json.dumps(data_with_time)
            )
            
            print(f"✅ Option chain stored in Redis for {symbol}")
        except Exception as e:
            print(f"❌ Error storing option chain: {e}")
    
    def _get_previous_option_chain(self, symbol):
        """Get previous option chain data for comparison"""
        if not self.redis_client:
            return None
        
        try:
            key = f"{self.option_chain_cache_key}:{symbol}"
            data = self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"❌ Error retrieving previous option chain: {e}")
            return None
    
    def _compare_option_chains(self, symbol, current_options, previous_options):
        """Compare current and previous option chain data"""
        if not previous_options or not current_options:
            return {}
        
        try:
            curr_data = current_options
            prev_data = previous_options.get("data", {})
            
            comparison = {
                "pcr_change": curr_data.get("pcr_ratio", 0) - prev_data.get("pcr_ratio", 0),
                "call_oi_change": curr_data.get("total_call_oi", 0) - prev_data.get("total_call_oi", 0),
                "put_oi_change": curr_data.get("total_put_oi", 0) - prev_data.get("total_put_oi", 0),
                "previous_pcr": prev_data.get("pcr_ratio", 0),
                "current_pcr": curr_data.get("pcr_ratio", 0),
                "timestamp": previous_options.get("timestamp")
            }
            
            return comparison
        except Exception as e:
            print(f"❌ Error comparing option chains: {e}")
            return {}

    def fetch_candlestick_data(self, symbol, count=100):
        """Fetch historical candlestick data from Deribit"""
        try:
            instrument = f"{symbol}-PERPETUAL"
            endpoint = f"{DERIBIT_API_URL}/get_tradingview_chart_data"
            
            params = {
                "instrument_name": instrument,
                "resolution": self.timeframe,
                "start_timestamp": int(time.time() * 1000) - (count * 30 * 60 * 1000),
                "end_timestamp": int(time.time() * 1000)
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("result"):
                result = data["result"]
                candles = []
                for i in range(len(result["ticks"])):
                    candles.append({
                        "timestamp": result["ticks"][i],
                        "open": result["open"][i],
                        "high": result["high"][i],
                        "low": result["low"][i],
                        "close": result["close"][i],
                        "volume": result["volume"][i]
                    })
                return candles[-100:]
            return None
            
        except Exception as e:
            print(f"❌ Error fetching candlestick data for {symbol}: {e}")
            return None
    
    def fetch_option_chain_data(self, symbol):
        """Fetch option chain data from Deribit"""
        try:
            currency = symbol.replace("-PERPETUAL", "")
            endpoint = f"{DERIBIT_API_URL}/get_book_summary_by_currency"
            
            params = {
                "currency": currency,
                "kind": "option"
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("result"):
                options = data["result"]
                
                call_oi = sum(opt.get("open_interest", 0) for opt in options if "C" in opt["instrument_name"])
                put_oi = sum(opt.get("open_interest", 0) for opt in options if "P" in opt["instrument_name"])
                
                pcr = put_oi / call_oi if call_oi > 0 else 0
                
                strikes_data = {}
                for opt in options[:50]:
                    strike = opt.get("instrument_name", "").split("-")[2] if len(opt.get("instrument_name", "").split("-")) > 2 else None
                    if strike and strike.isdigit():
                        if strike not in strikes_data:
                            strikes_data[strike] = {"call_oi": 0, "put_oi": 0}
                        if "C" in opt["instrument_name"]:
                            strikes_data[strike]["call_oi"] += opt.get("open_interest", 0)
                        else:
                            strikes_data[strike]["put_oi"] += opt.get("open_interest", 0)
                
                option_data = {
                    "pcr_ratio": round(pcr, 2),
                    "total_call_oi": call_oi,
                    "total_put_oi": put_oi,
                    "significant_strikes": dict(sorted(strikes_data.items(), 
                                                      key=lambda x: x[1]["call_oi"] + x[1]["put_oi"], 
                                                      reverse=True)[:10])
                }
                
                # Store in Redis for comparison
                self._store_option_chain(symbol, option_data)
                
                return option_data
            return None
            
        except Exception as e:
            print(f"❌ Error fetching option chain for {symbol}: {e}")
            return None
    
    def calculate_ema(self, candles, period):
        """Calculate Exponential Moving Average"""
        prices = [c["close"] for c in candles]
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return round(ema, 2)
    
    def analyze_candlestick_patterns(self, candles):
        """Identify candlestick patterns (last 3 candles)"""
        if len(candles) < 3:
            return []
        
        patterns = []
        last = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3]
        
        # Bullish patterns
        if last["close"] > last["open"] and prev["close"] < prev["open"]:
            body_ratio = abs(last["close"] - last["open"]) / abs(prev["close"] - prev["open"]) if abs(prev["close"] - prev["open"]) > 0 else 0
            if body_ratio > 1.5:
                patterns.append("Bullish Engulfing")
        
        # Bearish patterns
        if last["close"] < last["open"] and prev["close"] > prev["open"]:
            body_ratio = abs(last["close"] - last["open"]) / abs(prev["close"] - prev["open"]) if abs(prev["close"] - prev["open"]) > 0 else 0
            if body_ratio > 1.5:
                patterns.append("Bearish Engulfing")
        
        # Doji
        body = abs(last["close"] - last["open"])
        range_size = last["high"] - last["low"]
        if range_size > 0 and body / range_size < 0.1:
            patterns.append("Doji")
        
        # Hammer
        if last["close"] > last["open"]:
            lower_wick = last["open"] - last["low"]
            body_size = last["close"] - last["open"]
            if body_size > 0 and lower_wick > body_size * 2:
                patterns.append("Hammer (Bullish)")
        
        return patterns
    
    def identify_support_resistance(self, candles):
        """Identify key support and resistance levels"""
        highs = [c["high"] for c in candles[-50:]]
        lows = [c["low"] for c in candles[-50:]]
        
        resistance = max(highs)
        support = min(lows)
        
        current_price = candles[-1]["close"]
        
        return {
            "resistance": round(resistance, 2),
            "support": round(support, 2),
            "current": round(current_price, 2),
            "distance_to_resistance": round((resistance - current_price) / current_price * 100, 2),
            "distance_to_support": round((current_price - support) / current_price * 100, 2)
        }
    
    def prepare_analysis_data(self, symbol):
        """Prepare complete data for AI analysis (80% bot work)"""
        print(f"\n{'='*50}")
        print(f"Analyzing {symbol}...")
        print(f"{'='*50}")
        
        # Fetch data
        candles = self.fetch_candlestick_data(symbol)
        options = self.fetch_option_chain_data(symbol)
        
        if not candles:
            print(f"❌ Failed to fetch candle data for {symbol}")
            return None
        
        # Get previous option chain for comparison
        previous_options = self._get_previous_option_chain(symbol)
        option_comparison = self._compare_option_chains(symbol, options, previous_options) if options else {}
        
        # Calculate EMAs
        emas = {}
        for period in self.ema_periods:
            ema = self.calculate_ema(candles, period)
            if ema:
                emas[f"EMA_{period}"] = ema
        
        # Get patterns and levels
        patterns = self.analyze_candlestick_patterns(candles)
        sr_levels = self.identify_support_resistance(candles)
        
        # Recent price action
        last_5_candles = candles[-5:]
        
        analysis_data = {
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": candles[-1]["close"],
            "emas": emas,
            "candlestick_patterns": patterns,
            "support_resistance": sr_levels,
            "option_chain": options,
            "option_chain_comparison": option_comparison,
            "recent_candles": [
                {
                    "time": datetime.fromtimestamp(c["timestamp"]/1000).strftime("%H:%M"),
                    "o": c["open"],
                    "h": c["high"],
                    "l": c["low"],
                    "c": c["close"],
                    "v": c["volume"]
                } for c in last_5_candles
            ],
            "volume_trend": "Increasing" if last_5_candles[-1]["volume"] > sum(c["volume"] for c in last_5_candles[:-1])/4 else "Decreasing"
        }
        
        print(f"✅ Data collected for {symbol}")
        print(f"Current Price: ${analysis_data['current_price']}")
        print(f"Patterns Found: {', '.join(patterns) if patterns else 'None'}")
        print(f"PCR Ratio: {options['pcr_ratio'] if options else 'N/A'}")
        if option_comparison:
            print(f"PCR Change: {option_comparison.get('pcr_change', 0):+.2f}")
        
        # Manage memory
        self._manage_redis_memory()
        
        return analysis_data
    
    def _is_high_probability_signal(self, analysis_data):
        """Check if technical setup is high probability before AI analysis"""
        checks = []
        
        # EMA alignment check
        emas = analysis_data['emas']
        if emas:
            ema_9 = emas.get('EMA_9', 0)
            ema_21 = emas.get('EMA_21', 0)
            ema_50 = emas.get('EMA_50', 0)
            current = analysis_data['current_price']
            
            # Bullish alignment: price > EMA9 > EMA21 > EMA50
            bullish_aligned = current > ema_9 > ema_21 > ema_50
            # Bearish alignment: price < EMA9 < EMA21 < EMA50
            bearish_aligned = current < ema_9 < ema_21 < ema_50
            
            checks.append(bullish_aligned or bearish_aligned)
        
        # Pattern check
        patterns = analysis_data.get('candlestick_patterns', [])
        has_pattern = len(patterns) > 0
        checks.append(has_pattern)
        
        # Volume check
        volume_trend = analysis_data.get('volume_trend') == "Increasing"
        checks.append(volume_trend)
        
        # Support/Resistance proximity check
        sr = analysis_data['support_resistance']
        near_support_or_resistance = (sr['distance_to_support'] < 2) or (sr['distance_to_resistance'] < 2)
        checks.append(near_support_or_resistance)
        
        # Option chain sentiment check
        option_chain = analysis_data.get('option_chain')
        if option_chain:
            pcr = option_chain.get('pcr_ratio', 0)
            # Strong sentiment if PCR is extreme
            strong_sentiment = pcr > 1.5 or pcr < 0.7
            checks.append(strong_sentiment)
        else:
            checks.append(True)  # Don't penalize if no data
        
        # Need at least 3 out of 5 checks to pass
        probability = sum(checks) / len(checks)
        is_high_prob = sum(checks) >= 3
        
        return is_high_prob, probability

    def get_ai_analysis(self, analysis_data, is_high_prob=True):
        """Get AI analysis from DeepSeek V3 (20% AI work)"""
        try:
            option_comparison_str = ""
            if analysis_data.get('option_chain_comparison'):
                comp = analysis_data['option_chain_comparison']
                option_comparison_str = f"""
OPTION CHAIN COMPARISON (30 min ago):
- Previous PCR: {comp.get('previous_pcr', 'N/A')}
- Current PCR: {comp.get('current_pcr', 'N/A')}
- PCR Change: {comp.get('pcr_change', 0):+.2f}
- Call OI Change: {comp.get('call_oi_change', 0):+,.0f}
- Put OI Change: {comp.get('put_oi_change', 0):+,.0f}
"""
            
            prompt = f"""
You are an expert crypto trader analyzing {analysis_data['symbol']} for HIGH-PROBABILITY trading signals.

⚠️ IMPORTANT: This setup has already passed initial technical filters. 
Only provide BUY/SELL if you confirm HIGH confidence with multiple converging factors.
Otherwise, return NEUTRAL.

MARKET DATA:
- Current Price: ${analysis_data['current_price']}
- EMAs: {json.dumps(analysis_data['emas'], indent=2)}
- Support: ${analysis_data['support_resistance']['support']} ({analysis_data['support_resistance']['distance_to_support']}% away)
- Resistance: ${analysis_data['support_resistance']['resistance']} ({analysis_data['support_resistance']['distance_to_resistance']}% away)
- Candlestick Patterns: {', '.join(analysis_data['candlestick_patterns']) if analysis_data['candlestick_patterns'] else 'None'}
- Volume Trend: {analysis_data['volume_trend']}

CURRENT OPTION DATA:
{json.dumps(analysis_data['option_chain'], indent=2) if analysis_data['option_chain'] else 'Not available'}

{option_comparison_str}

RECENT PRICE ACTION (Last 5 candles):
{json.dumps(analysis_data['recent_candles'], indent=2)}

ANALYSIS REQUIREMENTS:
1. Analyze EMA alignment and price position
2. Evaluate candlestick patterns significance
3. Check support/resistance and option flow
4. Compare with 30-min-ago option data for shifts
5. Assess volume and momentum
6. Look for confluence of multiple factors

Provide EXACT format:

SIGNAL: [BUY/SELL/NEUTRAL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
ENTRY: $[price]
TARGET: $[price]
STOP_LOSS: $[price]
REASONING: [2-3 sentences max]
RISK_REWARD: [ratio]

Be STRICT - only HIGH confidence signals for BUY/SELL when multiple factors align.
"""
            
            print(f"\n🤖 AI analyzing {analysis_data['symbol']} (High Probability Pre-Check: {'PASS ✅' if is_high_prob else 'FAIL ❌'})")
            
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a professional crypto trader providing ONLY high-probability trading signals. Be conservative."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.2  # More conservative
            )
            
            ai_response = response.choices[0].message.content
            print(f"✅ AI analysis completed")
            
            return ai_response
            
        except Exception as e:
            print(f"❌ Error getting AI analysis: {e}")
            return None
    
    async def send_telegram_alert(self, symbol, analysis_data, ai_analysis):
        """Send trading signal to Telegram"""
        try:
            option_comp_str = ""
            if analysis_data.get('option_chain_comparison'):
                comp = analysis_data['option_chain_comparison']
                option_comp_str = f"""
<b>📊 Option Change (30 min):</b>
• PCR: {comp.get('previous_pcr', 0):.2f} → {comp.get('current_pcr', 0):.2f} ({comp.get('pcr_change', 0):+.2f})
• Call OI: {comp.get('call_oi_change', 0):+,.0f}
• Put OI: {comp.get('put_oi_change', 0):+,.0f}
"""
            
            message = f"""
🚨 <b>CRYPTO TRADING SIGNAL</b> 🚨

<b>Symbol:</b> {symbol}
<b>Time:</b> {analysis_data['timestamp']}
<b>Current Price:</b> ${analysis_data['current_price']}

<b>🤖 AI ANALYSIS:</b>
<code>{ai_analysis}</code>

<b>📊 TECHNICAL DATA:</b>
• EMAs: {', '.join([f'{k}: ${v}' for k, v in analysis_data['emas'].items()])}
• Support: ${analysis_data['support_resistance']['support']}
• Resistance: ${analysis_data['support_resistance']['resistance']}
• Patterns: {', '.join(analysis_data['candlestick_patterns']) if analysis_data['candlestick_patterns'] else 'None'}

<b>📈 OPTION DATA:</b>
• PCR Ratio: {analysis_data['option_chain']['pcr_ratio'] if analysis_data['option_chain'] else 'N/A'}
• Volume Trend: {analysis_data['volume_trend']}

{option_comp_str}

⚠️ <i>Trade at your own risk. Not financial advice.</i>
"""
            
            await telegram_bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            
            print(f"✅ Alert sent to Telegram for {symbol}")
            return True
            
        except TelegramError as e:
            print(f"❌ Telegram error: {e}")
            return False
        except Exception as e:
            print(f"❌ Error sending alert: {e}")
            return False
    
    def should_send_alert(self, ai_analysis):
        """Check if signal is high probability and should be sent"""
        if not ai_analysis:
            return False
        
        analysis_upper = ai_analysis.upper()
        
        # STRICT: Only BUY/SELL with HIGH confidence
        has_buy_signal = "SIGNAL: BUY" in analysis_upper
        has_sell_signal = "SIGNAL: SELL" in analysis_upper
        has_signal = has_buy_signal or has_sell_signal
        has_high_confidence = "CONFIDENCE: HIGH" in analysis_upper
        
        should_send = has_signal and has_high_confidence
        
        if not should_send:
            signal_type = "NEUTRAL" if "SIGNAL: NEUTRAL" in analysis_upper else "UNKNOWN"
            confidence = "N/A"
            if "CONFIDENCE: HIGH" in analysis_upper:
                confidence = "HIGH"
            elif "CONFIDENCE: MEDIUM" in analysis_upper:
                confidence = "MEDIUM"
            elif "CONFIDENCE: LOW" in analysis_upper:
                confidence = "LOW"
            print(f"⏭️  Signal: {signal_type}, Confidence: {confidence} - Not sending alert")
        
        return should_send
    
    async def run_analysis_cycle(self):
        """Run one complete analysis cycle for all symbols"""
        print(f"\n{'='*60}")
        print(f"🔄 Starting Analysis Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        for symbol in self.symbols:
            try:
                # 80% work - bot does data collection
                analysis_data = self.prepare_analysis_data(symbol)
                
                if not analysis_data:
                    continue
                
                # Pre-check: Is this high probability setup?
                is_high_prob, prob_score = self._is_high_probability_signal(analysis_data)
                print(f"📊 Technical Probability Score: {prob_score:.0%}")
                
                if not is_high_prob:
                    print(f"⏭️  {symbol} - Low probability setup, skipping AI analysis")
                    await asyncio.sleep(1)
                    continue
                
                # 20% work - AI does analysis for high prob signals only
                ai_analysis = self.get_ai_analysis(analysis_data, is_high_prob)
                
                if not ai_analysis:
                    continue
                
                # Send alert only for HIGH confidence signals
                if self.should_send_alert(ai_analysis):
                    print(f"\n🎯 HIGH PROBABILITY SIGNAL DETECTED for {symbol}!")
                    await self.send_telegram_alert(symbol, analysis_data, ai_analysis)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
                continue
        
        print(f"\n✅ Analysis cycle completed")
        print(f"{'='*60}\n")
    
    async def start(self):
        """Start the bot with continuous monitoring"""
        print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║      🚀 CRYPTO TRADING BOT WITH DEEPSEEK V3 🚀          ║
║           + REDIS MEMORY + OPTION TRACKING              ║
║                                                          ║
║  Analyzing: BTC & ETH                                    ║
║  Timeframe: 30 minutes                                   ║
║  Signals: HIGH-PROBABILITY only (Pre-filtered + AI)     ║
║  Memory: Redis with auto-cleanup                         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
        """)
        
        # Test Telegram connection
        try:
            await telegram_bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text="✅ Bot started with Redis caching! Monitoring BTC & ETH for HIGH-PROBABILITY signals...",
                parse_mode='HTML'
            )
            print("✅ Telegram connection verified\n")
        except Exception as e:
            print(f"❌ Telegram connection failed: {e}\n")
        
        # Continuous monitoring loop
        while True:
            try:
                await self.run_analysis_cycle()
                
                print(f"⏰ Next scan in {SCAN_INTERVAL//60} minutes...")
                await asyncio.sleep(SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                print("⏰ Retrying in 5 minutes...")
                await asyncio.sleep(300)


def main():
    """Main entry point"""
    if not DEEPSEEK_API_KEY:
        print("❌ ERROR: DEEPSEEK_API_KEY not found in .env file")
        return
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ ERROR: Telegram credentials not found in .env file")
        return
    
    analyzer = CryptoAnalyzer()
    asyncio.run(analyzer.start())


if __name__ == "__main__":
    main()
