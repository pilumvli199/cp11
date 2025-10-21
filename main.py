import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from telegram import Bot, InputFile
from telegram.error import TelegramError
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import traceback
from dotenv import load_dotenv

load_dotenv()

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

COINS = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'DOGE', 'TRX', 'ADA', 'AVAX', 'LINK']
TIMEFRAMES = ['1h', '4h', '1d']
SCAN_INTERVAL = 3600  # 1 hour in seconds

class BinanceAPI:
    """Binance Futures API - More reliable and supports all coins"""
    BASE_URL = "https://fapi.binance.com/fapi/v1"
    
    @staticmethod
    async def get_candlestick_data(session, symbol, timeframe, limit=1000):
        """Fetch candlestick data from Binance Futures API"""
        url = f"{BinanceAPI.BASE_URL}/klines"
        
        params = {
            'symbol': f"{symbol}USDT",
            'interval': timeframe,
            'limit': min(limit, 1500)  # Binance max limit
        }
        
        try:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if len(data) > 0:
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 
                            'taker_buy_base', 'taker_buy_quote', 'ignore'
                        ])
                        
                        # Convert to proper types
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df['open'] = df['open'].astype(float)
                        df['high'] = df['high'].astype(float)
                        df['low'] = df['low'].astype(float)
                        df['close'] = df['close'].astype(float)
                        df['volume'] = df['volume'].astype(float)
                        
                        # Keep only needed columns
                        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                        
                        print(f"✓ Fetched {len(df)} candles for {symbol} {timeframe}")
                        return df
                    
        except asyncio.TimeoutError:
            print(f"⚠️ Timeout fetching {symbol} {timeframe}")
        except Exception as e:
            print(f"❌ Error fetching {symbol} {timeframe}: {str(e)}")
        
        return None


class DeribitAPI:
    """Deribit API - Only for BTC and ETH"""
    BASE_URL = "https://www.deribit.com/api/v2/public"
    
    @staticmethod
    async def get_candlestick_data(session, symbol, timeframe, limit=1000):
        """Fetch candlestick data from Deribit public API"""
        # Deribit only has BTC and ETH perpetuals
        if symbol not in ['BTC', 'ETH']:
            return None
            
        # Convert timeframe to minutes
        tf_map = {'1h': 60, '4h': 240, '1d': 1440}
        resolution = tf_map.get(timeframe, 60)
        
        # Calculate timestamps
        end_timestamp = int(datetime.now().timestamp() * 1000)
        start_timestamp = end_timestamp - (limit * resolution * 60 * 1000)
        
        # Deribit instrument naming
        instrument = f"{symbol}-PERPETUAL"
        url = f"{DeribitAPI.BASE_URL}/get_tradingview_chart_data"
        
        params = {
            'instrument_name': instrument,
            'resolution': resolution,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp
        }
        
        try:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'result' in data and data['result']['status'] == 'ok':
                        result = data['result']
                        df = pd.DataFrame({
                            'timestamp': result['ticks'],
                            'open': result['open'],
                            'high': result['high'],
                            'low': result['low'],
                            'close': result['close'],
                            'volume': result['volume']
                        })
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        print(f"✓ Fetched {len(df)} candles for {symbol} {timeframe} from Deribit")
                        return df
        except Exception as e:
            print(f"⚠️ Deribit error for {symbol} {timeframe}: {str(e)}")
        
        return None

class SMCAnalyzer:
    """Smart Money Concepts Analyzer"""
    
    @staticmethod
    def find_order_blocks(df, lookback=20):
        """Identify bullish and bearish order blocks"""
        order_blocks = {'bullish': [], 'bearish': []}
        
        for i in range(lookback, len(df) - 1):
            # Bullish Order Block: Strong buying candle followed by upward movement
            if df['close'].iloc[i] > df['open'].iloc[i]:
                body_size = df['close'].iloc[i] - df['open'].iloc[i]
                avg_body = abs(df['close'].iloc[i-lookback:i] - df['open'].iloc[i-lookback:i]).mean()
                
                if body_size > avg_body * 1.5:
                    order_blocks['bullish'].append({
                        'index': i,
                        'price': df['low'].iloc[i],
                        'high': df['high'].iloc[i],
                        'strength': body_size
                    })
            
            # Bearish Order Block: Strong selling candle followed by downward movement
            elif df['close'].iloc[i] < df['open'].iloc[i]:
                body_size = df['open'].iloc[i] - df['close'].iloc[i]
                avg_body = abs(df['close'].iloc[i-lookback:i] - df['open'].iloc[i-lookback:i]).mean()
                
                if body_size > avg_body * 1.5:
                    order_blocks['bearish'].append({
                        'index': i,
                        'price': df['high'].iloc[i],
                        'low': df['low'].iloc[i],
                        'strength': body_size
                    })
        
        return order_blocks
    
    @staticmethod
    def detect_bos_choch(df, swing_period=10):
        """Detect Break of Structure (BOS) and Change of Character (ChoCH)"""
        signals = []
        highs = df['high'].rolling(window=swing_period).max()
        lows = df['low'].rolling(window=swing_period).min()
        
        for i in range(swing_period, len(df) - 1):
            # Bullish BOS: Break above previous swing high
            if df['close'].iloc[i] > highs.iloc[i-1]:
                signals.append({'type': 'BOS_BULL', 'index': i, 'price': df['close'].iloc[i]})
            
            # Bearish BOS: Break below previous swing low
            if df['close'].iloc[i] < lows.iloc[i-1]:
                signals.append({'type': 'BOS_BEAR', 'index': i, 'price': df['close'].iloc[i]})
        
        return signals
    
    @staticmethod
    def find_fvg(df):
        """Find Fair Value Gaps"""
        fvgs = {'bullish': [], 'bearish': []}
        
        for i in range(2, len(df)):
            # Bullish FVG: Gap between candle[i-2] high and candle[i] low
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                fvgs['bullish'].append({
                    'index': i,
                    'top': df['low'].iloc[i],
                    'bottom': df['high'].iloc[i-2],
                    'size': gap_size
                })
            
            # Bearish FVG: Gap between candle[i-2] low and candle[i] high
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                fvgs['bearish'].append({
                    'index': i,
                    'top': df['low'].iloc[i-2],
                    'bottom': df['high'].iloc[i],
                    'size': gap_size
                })
        
        return fvgs
    
    @staticmethod
    def detect_candlestick_patterns(df):
        """Detect common candlestick patterns"""
        patterns = []
        
        for i in range(2, len(df)):
            # Bullish Engulfing
            if (df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i-1] < df['open'].iloc[i-1] and
                df['open'].iloc[i] < df['close'].iloc[i-1] and
                df['close'].iloc[i] > df['open'].iloc[i-1]):
                patterns.append({'type': 'Bullish_Engulfing', 'index': i})
            
            # Bearish Engulfing
            if (df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i-1] > df['open'].iloc[i-1] and
                df['open'].iloc[i] > df['close'].iloc[i-1] and
                df['close'].iloc[i] < df['open'].iloc[i-1]):
                patterns.append({'type': 'Bearish_Engulfing', 'index': i})
            
            # Hammer (Bullish)
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            lower_shadow = min(df['close'].iloc[i], df['open'].iloc[i]) - df['low'].iloc[i]
            upper_shadow = df['high'].iloc[i] - max(df['close'].iloc[i], df['open'].iloc[i])
            
            if lower_shadow > body * 2 and upper_shadow < body * 0.3:
                patterns.append({'type': 'Hammer', 'index': i})
            
            # Shooting Star (Bearish)
            if upper_shadow > body * 2 and lower_shadow < body * 0.3:
                patterns.append({'type': 'Shooting_Star', 'index': i})
        
        return patterns

class DeepSeekAnalyzer:
    """DeepSeek V3 API Integration"""
    
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    @staticmethod
    async def analyze_with_deepseek(session, coin, mtf_data, smc_data):
        """Send multi-timeframe data to DeepSeek V3 for analysis"""
        
        # Prepare analysis prompt
        prompt = f"""You are an expert cryptocurrency trader specializing in Smart Money Concepts (SMC). 

Analyze {coin}/USDT using the following multi-timeframe data and SMC indicators:

MULTI-TIMEFRAME DATA:
{mtf_data}

SMC ANALYSIS:
{smc_data}

Based on this information, provide:
1. **Overall Market Structure**: Bullish/Bearish/Neutral
2. **Key Support/Resistance Levels**
3. **Entry Signal**: BUY/SELL/WAIT with confidence level (0-100%)
4. **Entry Price Range**
5. **Stop Loss Level**
6. **Take Profit Targets** (TP1, TP2, TP3)
7. **Risk Assessment**: Low/Medium/High
8. **Key Reasoning** (2-3 sentences max)

Format your response as JSON:
{{
    "signal": "BUY/SELL/WAIT",
    "confidence": 85,
    "entry_range": [45000, 45500],
    "stop_loss": 44000,
    "take_profits": [46000, 47000, 48000],
    "risk": "Medium",
    "structure": "Bullish",
    "reasoning": "Strong bullish order block with FVG fill..."
}}"""

        headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': 'You are a professional crypto trader expert in Smart Money Concepts.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.3,
            'max_tokens': 1000
        }
        
        try:
            async with session.post(DeepSeekAnalyzer.API_URL, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    # Extract JSON from response
                    import json
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
        except Exception as e:
            print(f"DeepSeek API Error for {coin}: {str(e)}")
        
        return None

class ChartGenerator:
    """Generate beautiful candlestick charts"""
    
    @staticmethod
    def create_chart(df, coin, timeframe, smc_data, analysis):
        """Create PNG chart with white background and SMC indicators"""
        
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        ax.set_facecolor('white')
        
        # Plot candlesticks
        for i in range(len(df)):
            color = 'green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
            
            # Candle body
            body_height = abs(df['close'].iloc[i] - df['open'].iloc[i])
            body_bottom = min(df['open'].iloc[i], df['close'].iloc[i])
            
            ax.add_patch(patches.Rectangle(
                (i, body_bottom), 0.8, body_height,
                facecolor=color, edgecolor='black', linewidth=0.5
            ))
            
            # Wicks
            ax.plot([i + 0.4, i + 0.4], 
                   [df['low'].iloc[i], df['high'].iloc[i]], 
                   color='black', linewidth=1)
        
        # Plot Order Blocks
        if 'order_blocks' in smc_data:
            for ob in smc_data['order_blocks']['bullish'][-3:]:
                idx = ob['index']
                if idx < len(df):
                    ax.axhspan(ob['price'], ob['high'], 
                              alpha=0.2, color='green', label='Bullish OB')
            
            for ob in smc_data['order_blocks']['bearish'][-3:]:
                idx = ob['index']
                if idx < len(df):
                    ax.axhspan(ob['low'], ob['price'], 
                              alpha=0.2, color='red', label='Bearish OB')
        
        # Plot FVG
        if 'fvg' in smc_data:
            for fvg in smc_data['fvg']['bullish'][-2:]:
                idx = fvg['index']
                if idx < len(df):
                    ax.axhspan(fvg['bottom'], fvg['top'], 
                              alpha=0.15, color='blue', linestyle='--')
        
        # Add signal annotation
        if analysis and analysis['signal'] in ['BUY', 'SELL']:
            color = 'green' if analysis['signal'] == 'BUY' else 'red'
            y_pos = df['low'].min() if analysis['signal'] == 'BUY' else df['high'].max()
            
            ax.annotate(f"{analysis['signal']}\n{analysis['confidence']}%",
                       xy=(len(df)-10, y_pos),
                       xytext=(len(df)-20, y_pos),
                       fontsize=14, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # Styling
        ax.set_title(f'{coin}/USDT - {timeframe.upper()} | SMC Analysis', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Candles', fontsize=12)
        ax.set_ylabel('Price (USDT)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        fig.text(0.99, 0.01, f'Generated: {timestamp}', 
                ha='right', fontsize=8, color='gray')
        
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='white')
        buf.seek(0)
        plt.close()
        
        return buf

class TradingBot:
    """Main Trading Bot Controller"""
    
    def __init__(self):
        self.telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.is_running = False
    
    async def send_telegram_alert(self, coin, timeframe, analysis, chart_buffer):
        """Send trading alert to Telegram"""
        try:
            if analysis['signal'] == 'WAIT':
                return  # Don't send WAIT signals
            
            signal_emoji = "🟢" if analysis['signal'] == 'BUY' else "🔴"
            
            message = f"""
{signal_emoji} **{analysis['signal']} ALERT** {signal_emoji}

**Coin:** {coin}/USDT
**Timeframe:** {timeframe.upper()}
**Confidence:** {analysis['confidence']}%
**Market Structure:** {analysis['structure']}

**Entry Range:** ${analysis['entry_range'][0]:,.2f} - ${analysis['entry_range'][1]:,.2f}
**Stop Loss:** ${analysis['stop_loss']:,.2f}
**Take Profits:**
  TP1: ${analysis['take_profits'][0]:,.2f}
  TP2: ${analysis['take_profits'][1]:,.2f}
  TP3: ${analysis['take_profits'][2]:,.2f}

**Risk Level:** {analysis['risk']}

**Analysis:**
{analysis['reasoning']}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            """
            
            # Send chart
            chart_buffer.seek(0)
            await self.telegram_bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=InputFile(chart_buffer, filename=f'{coin}_{timeframe}.png'),
                caption=message,
                parse_mode='Markdown'
            )
            
            print(f"✅ Alert sent for {coin} {timeframe}: {analysis['signal']}")
            
        except TelegramError as e:
            print(f"Telegram Error: {str(e)}")
        except Exception as e:
            print(f"Error sending alert: {str(e)}")
    
    async def analyze_coin(self, session, coin, timeframe):
        """Analyze single coin on single timeframe"""
        try:
            # Try Binance first (more reliable and supports all coins)
            df = await BinanceAPI.get_candlestick_data(session, coin, timeframe, limit=1000)
            
            # Fallback to Deribit only for BTC/ETH if Binance fails
            if df is None and coin in ['BTC', 'ETH']:
                print(f"⚠️ Binance failed, trying Deribit for {coin}...")
                df = await DeribitAPI.get_candlestick_data(session, coin, timeframe, limit=1000)
            
            if df is None or len(df) < 100:
                print(f"⚠️ Insufficient data for {coin} {timeframe} (got {len(df) if df is not None else 0} candles)")
                return
            
            # Perform SMC Analysis
            order_blocks = SMCAnalyzer.find_order_blocks(df)
            bos_choch = SMCAnalyzer.detect_bos_choch(df)
            fvg = SMCAnalyzer.find_fvg(df)
            patterns = SMCAnalyzer.detect_candlestick_patterns(df)
            
            smc_data = {
                'order_blocks': order_blocks,
                'bos_choch': bos_choch,
                'fvg': fvg,
                'patterns': patterns
            }
            
            # Prepare data for DeepSeek
            recent_data = df.tail(50).to_dict('records')
            mtf_summary = {
                'current_price': float(df['close'].iloc[-1]),
                'change_24h': float((df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100),
                'high_24h': float(df['high'].tail(24).max()),
                'low_24h': float(df['low'].tail(24).min()),
                'volume': float(df['volume'].tail(24).sum())
            }
            
            smc_summary = {
                'bullish_order_blocks': len(order_blocks['bullish']),
                'bearish_order_blocks': len(order_blocks['bearish']),
                'bullish_fvg': len(fvg['bullish']),
                'bearish_fvg': len(fvg['bearish']),
                'recent_patterns': [p['type'] for p in patterns[-5:]],
                'bos_signals': len([s for s in bos_choch if 'BULL' in s['type']])
            }
            
            # Get DeepSeek analysis
            analysis = await DeepSeekAnalyzer.analyze_with_deepseek(
                session, coin, str(mtf_summary), str(smc_summary)
            )
            
            if analysis:
                # Generate chart
                chart_buffer = ChartGenerator.create_chart(
                    df.tail(100), coin, timeframe, smc_data, analysis
                )
                
                # Send alert if signal is BUY or SELL
                await self.send_telegram_alert(coin, timeframe, analysis, chart_buffer)
            
        except Exception as e:
            print(f"❌ Error analyzing {coin} {timeframe}: {str(e)}")
            traceback.print_exc()
    
    async def scan_all_coins(self):
        """Scan all coins across all timeframes"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for coin in COINS:
                for timeframe in TIMEFRAMES:
                    task = self.analyze_coin(session, coin, timeframe)
                    tasks.append(task)
                    await asyncio.sleep(0.5)  # Rate limiting
            
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def run(self):
        """Main bot loop"""
        self.is_running = True
        print("🚀 Trading Bot Started!")
        print(f"📊 Scanning: {', '.join(COINS)}")
        print(f"⏱️ Timeframes: {', '.join(TIMEFRAMES)}")
        print(f"🔄 Scan Interval: {SCAN_INTERVAL}s (1 hour)\n")
        
        while self.is_running:
            try:
                print(f"\n{'='*50}")
                print(f"🔍 Starting scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*50}\n")
                
                await self.scan_all_coins()
                
                print(f"\n✅ Scan completed. Next scan in {SCAN_INTERVAL}s")
                await asyncio.sleep(SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                self.is_running = False
                break
            except Exception as e:
                print(f"\n❌ Critical error: {str(e)}")
                traceback.print_exc()
                await asyncio.sleep(60)  # Wait before retry

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.run())
