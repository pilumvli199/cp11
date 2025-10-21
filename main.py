import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from telegram import Bot
from telegram.request import HTTPXRequest
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
TIMEFRAMES = ['1d', '4h', '1h']  # Higher to lower for proper analysis
SCAN_INTERVAL = 3600  # 1 hour in seconds

class BinanceAPI:
    """Binance Futures API"""
    BASE_URL = "https://fapi.binance.com/fapi/v1"
    
    @staticmethod
    async def get_candlestick_data(session, symbol, timeframe, limit=1000):
        """Fetch candlestick data from Binance Futures API"""
        url = f"{BinanceAPI.BASE_URL}/klines"
        
        params = {
            'symbol': f"{symbol}USDT",
            'interval': timeframe,
            'limit': min(limit, 1500)
        }
        
        try:
            async with session.get(url, params=params, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if len(data) > 0:
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 
                            'taker_buy_base', 'taker_buy_quote', 'ignore'
                        ])
                        
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df['open'] = df['open'].astype(float)
                        df['high'] = df['high'].astype(float)
                        df['low'] = df['low'].astype(float)
                        df['close'] = df['close'].astype(float)
                        df['volume'] = df['volume'].astype(float)
                        
                        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                        return df
                    
        except Exception as e:
            print(f"❌ Error fetching {symbol} {timeframe}: {str(e)}")
        
        return None

class SMCAnalyzer:
    """Smart Money Concepts Analyzer with Multi-Timeframe Support"""
    
    @staticmethod
    def find_order_blocks(df, lookback=20):
        """Identify bullish and bearish order blocks"""
        order_blocks = {'bullish': [], 'bearish': []}
        
        for i in range(lookback, len(df) - 1):
            # Bullish Order Block
            if df['close'].iloc[i] > df['open'].iloc[i]:
                body_size = df['close'].iloc[i] - df['open'].iloc[i]
                avg_body = abs(df['close'].iloc[i-lookback:i] - df['open'].iloc[i-lookback:i]).mean()
                
                if body_size > avg_body * 1.5:
                    order_blocks['bullish'].append({
                        'price': float(df['low'].iloc[i]),
                        'high': float(df['high'].iloc[i]),
                        'strength': float(body_size),
                        'timestamp': str(df['timestamp'].iloc[i])
                    })
            
            # Bearish Order Block
            elif df['close'].iloc[i] < df['open'].iloc[i]:
                body_size = df['open'].iloc[i] - df['close'].iloc[i]
                avg_body = abs(df['close'].iloc[i-lookback:i] - df['open'].iloc[i-lookback:i]).mean()
                
                if body_size > avg_body * 1.5:
                    order_blocks['bearish'].append({
                        'price': float(df['high'].iloc[i]),
                        'low': float(df['low'].iloc[i]),
                        'strength': float(body_size),
                        'timestamp': str(df['timestamp'].iloc[i])
                    })
        
        return order_blocks
    
    @staticmethod
    def detect_bos_choch(df, swing_period=10):
        """Detect Break of Structure and Change of Character"""
        signals = []
        highs = df['high'].rolling(window=swing_period).max()
        lows = df['low'].rolling(window=swing_period).min()
        
        for i in range(swing_period, len(df) - 1):
            if df['close'].iloc[i] > highs.iloc[i-1]:
                signals.append({
                    'type': 'BOS_BULL',
                    'price': float(df['close'].iloc[i]),
                    'timestamp': str(df['timestamp'].iloc[i])
                })
            
            if df['close'].iloc[i] < lows.iloc[i-1]:
                signals.append({
                    'type': 'BOS_BEAR',
                    'price': float(df['close'].iloc[i]),
                    'timestamp': str(df['timestamp'].iloc[i])
                })
        
        return signals
    
    @staticmethod
    def find_fvg(df):
        """Find Fair Value Gaps"""
        fvgs = {'bullish': [], 'bearish': []}
        
        for i in range(2, len(df)):
            # Bullish FVG
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                fvgs['bullish'].append({
                    'top': float(df['low'].iloc[i]),
                    'bottom': float(df['high'].iloc[i-2]),
                    'size': float(gap_size)
                })
            
            # Bearish FVG
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                fvgs['bearish'].append({
                    'top': float(df['low'].iloc[i-2]),
                    'bottom': float(df['high'].iloc[i]),
                    'size': float(gap_size)
                })
        
        return fvgs
    
    @staticmethod
    def find_support_resistance(df, window=20):
        """Find key support and resistance levels"""
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(float(df['high'].iloc[i]))
            if df['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(float(df['low'].iloc[i]))
        
        # Get recent unique levels
        resistance = sorted(list(set(resistance_levels)))[-5:] if resistance_levels else []
        support = sorted(list(set(support_levels)))[-5:] if support_levels else []
        
        return {'resistance': resistance, 'support': support}
    
    @staticmethod
    def analyze_trend(df):
        """Determine overall trend"""
        # Simple moving averages
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        current_price = df['close'].iloc[-1]
        sma20 = df['sma20'].iloc[-1]
        sma50 = df['sma50'].iloc[-1]
        
        if current_price > sma20 > sma50:
            return 'BULLISH'
        elif current_price < sma20 < sma50:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

class DeepSeekAnalyzer:
    """DeepSeek V3 API with Multi-Timeframe Analysis"""
    
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    @staticmethod
    async def analyze_multi_timeframe(session, coin, tf_data):
        """Analyze using combined multi-timeframe data"""
        
        prompt = f"""You are a professional crypto trader using Smart Money Concepts and Multi-Timeframe Analysis.

Analyze {coin}/USDT across multiple timeframes:

🕐 DAILY TIMEFRAME (1D):
- Current Price: ${tf_data['1d']['current_price']:,.2f}
- Trend: {tf_data['1d']['trend']}
- Support Levels: {tf_data['1d']['support']}
- Resistance Levels: {tf_data['1d']['resistance']}
- Bullish Order Blocks: {len(tf_data['1d']['order_blocks']['bullish'])}
- Bearish Order Blocks: {len(tf_data['1d']['order_blocks']['bearish'])}
- BOS Signals: {len(tf_data['1d']['bos_signals'])}

⏰ 4-HOUR TIMEFRAME (4H):
- Trend: {tf_data['4h']['trend']}
- Support Levels: {tf_data['4h']['support']}
- Resistance Levels: {tf_data['4h']['resistance']}
- Bullish FVG: {len(tf_data['4h']['fvg']['bullish'])}
- Bearish FVG: {len(tf_data['4h']['fvg']['bearish'])}
- Recent Order Blocks: Bull={len(tf_data['4h']['order_blocks']['bullish'])}, Bear={len(tf_data['4h']['order_blocks']['bearish'])}

⏱️ 1-HOUR TIMEFRAME (1H):
- Current Price: ${tf_data['1h']['current_price']:,.2f}
- Trend: {tf_data['1h']['trend']}
- Recent Highs: ${tf_data['1h']['high_24h']:,.2f}
- Recent Lows: ${tf_data['1h']['low_24h']:,.2f}
- Volume Trend: {tf_data['1h']['volume_trend']}

MULTI-TIMEFRAME ANALYSIS RULES:
1. Use 1D for overall market structure and major trend
2. Use 4H for intermediate levels and entry zone identification
3. Use 1H for precise entry/exit timing and targets
4. Only give BUY signal if: 1D bullish + 4H setup ready + 1H confirms entry
5. Only give SELL signal if: 1D bearish + 4H setup ready + 1H confirms entry
6. Otherwise signal is WAIT

Provide analysis in JSON format:
{{
    "signal": "BUY/SELL/WAIT",
    "confidence": 0-100,
    "timeframe_alignment": "ALIGNED/PARTIAL/CONFLICTING",
    "daily_structure": "Bullish/Bearish/Neutral",
    "4h_setup": "Ready/Forming/Weak",
    "1h_entry": "Confirmed/Wait/Invalid",
    "entry_range": [lower, upper],
    "stop_loss": price,
    "take_profits": [tp1, tp2, tp3],
    "risk_reward": "1:3",
    "key_levels": {{
        "support": [s1, s2],
        "resistance": [r1, r2]
    }},
    "reasoning": "2-3 sentence analysis explaining the multi-timeframe alignment"
}}

Be strict: Only give actionable BUY/SELL when all timeframes align properly."""

        headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': 'You are a professional trader expert in Smart Money Concepts and Multi-Timeframe Analysis.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.2,
            'max_tokens': 1500
        }
        
        try:
            async with session.post(DeepSeekAnalyzer.API_URL, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    
                    import json
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                else:
                    print(f"⚠️ DeepSeek API error: {response.status}")
                    
        except asyncio.TimeoutError:
            print(f"⚠️ DeepSeek timeout for {coin}")
        except Exception as e:
            print(f"❌ DeepSeek error for {coin}: {str(e)}")
        
        return None

class ChartGenerator:
    """Generate multi-timeframe charts"""
    
    @staticmethod
    def create_mtf_chart(coin, tf_data, analysis):
        """Create combined multi-timeframe chart"""
        
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
        
        timeframes = ['1d', '4h', '1h']
        titles = ['Daily (1D) - Market Structure', '4-Hour (4H) - Entry Setup', '1-Hour (1H) - Timing']
        
        for idx, (tf, title) in enumerate(zip(timeframes, titles)):
            ax = fig.add_subplot(gs[idx])
            df = tf_data[tf]['dataframe'].tail(100)
            
            # Plot candlesticks
            for i in range(len(df)):
                color = 'green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
                
                body_height = abs(df['close'].iloc[i] - df['open'].iloc[i])
                body_bottom = min(df['open'].iloc[i], df['close'].iloc[i])
                
                ax.add_patch(patches.Rectangle(
                    (i, body_bottom), 0.8, body_height,
                    facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8
                ))
                
                ax.plot([i + 0.4, i + 0.4], 
                       [df['low'].iloc[i], df['high'].iloc[i]], 
                       color='black', linewidth=1, alpha=0.6)
            
            # Plot support/resistance
            if 'support' in tf_data[tf] and tf_data[tf]['support']:
                for level in tf_data[tf]['support'][-2:]:
                    ax.axhline(y=level, color='green', linestyle='--', alpha=0.3, linewidth=1.5)
            
            if 'resistance' in tf_data[tf] and tf_data[tf]['resistance']:
                for level in tf_data[tf]['resistance'][-2:]:
                    ax.axhline(y=level, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
            
            ax.set_title(f'{title} | Trend: {tf_data[tf]["trend"]}', 
                        fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel('Price (USDT)', fontsize=10)
            ax.grid(True, alpha=0.2, linestyle=':')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Add signal annotation
        if analysis and analysis['signal'] in ['BUY', 'SELL']:
            color = 'green' if analysis['signal'] == 'BUY' else 'red'
            signal_text = f"""
{analysis['signal']} SIGNAL
Confidence: {analysis['confidence']}%
TF Alignment: {analysis['timeframe_alignment']}
Risk:Reward = {analysis.get('risk_reward', 'N/A')}
"""
            fig.text(0.98, 0.5, signal_text,
                    transform=fig.transFigure,
                    fontsize=11, fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
                    verticalalignment='center',
                    horizontalalignment='right')
        
        plt.suptitle(f'{coin}/USDT - Multi-Timeframe SMC Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        fig.text(0.99, 0.01, f'Generated: {timestamp}', 
                ha='right', fontsize=8, color='gray')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='white', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf

class TradingBot:
    """Main Trading Bot with Multi-Timeframe Analysis"""
    
    def __init__(self):
        # Configure Telegram with larger connection pool
        request = HTTPXRequest(
            connection_pool_size=20,
            connect_timeout=30,
            read_timeout=30,
            write_timeout=30,
            pool_timeout=30
        )
        self.telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN, request=request)
        self.is_running = False
        self.alert_semaphore = asyncio.Semaphore(3)  # Max 3 parallel alerts
    
    async def send_telegram_alert(self, coin, analysis, chart_buffer):
        """Send trading alert with rate limiting"""
        async with self.alert_semaphore:
            try:
                if analysis['signal'] == 'WAIT':
                    return
                
                signal_emoji = "🟢" if analysis['signal'] == 'BUY' else "🔴"
                
                message = f"""
{signal_emoji} **{analysis['signal']} ALERT** {signal_emoji}

**Coin:** {coin}/USDT
**Confidence:** {analysis['confidence']}%
**Timeframe Alignment:** {analysis['timeframe_alignment']}

📊 **Multi-Timeframe Structure:**
• Daily: {analysis['daily_structure']}
• 4H Setup: {analysis['4h_setup']}
• 1H Entry: {analysis['1h_entry']}

💰 **Trade Setup:**
**Entry:** ${analysis['entry_range'][0]:,.2f} - ${analysis['entry_range'][1]:,.2f}
**Stop Loss:** ${analysis['stop_loss']:,.2f}
**Targets:**
  TP1: ${analysis['take_profits'][0]:,.2f}
  TP2: ${analysis['take_profits'][1]:,.2f}
  TP3: ${analysis['take_profits'][2]:,.2f}

**Risk:Reward:** {analysis.get('risk_reward', 'N/A')}

📈 **Key Levels:**
Support: {', '.join([f'${s:,.0f}' for s in analysis['key_levels']['support']])}
Resistance: {', '.join([f'${r:,.0f}' for r in analysis['key_levels']['resistance']])}

**Analysis:**
{analysis['reasoning']}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
                """
                
                chart_buffer.seek(0)
                await self.telegram_bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=chart_buffer,
                    caption=message,
                    parse_mode='Markdown'
                )
                
                print(f"✅ Alert sent for {coin}: {analysis['signal']} ({analysis['confidence']}%)")
                await asyncio.sleep(1)  # Rate limiting
                
            except TelegramError as e:
                print(f"⚠️ Telegram error for {coin}: {str(e)}")
            except Exception as e:
                print(f"❌ Error sending alert for {coin}: {str(e)}")
    
    async def analyze_coin_mtf(self, session, coin):
        """Analyze single coin across all timeframes"""
        try:
            print(f"\n📊 Analyzing {coin} across multiple timeframes...")
            
            # Fetch data for all timeframes
            tf_data = {}
            for tf in TIMEFRAMES:
                df = await BinanceAPI.get_candlestick_data(session, coin, tf, limit=1000)
                
                if df is None or len(df) < 100:
                    print(f"⚠️ Insufficient data for {coin} {tf}")
                    return
                
                # Perform SMC analysis
                order_blocks = SMCAnalyzer.find_order_blocks(df)
                bos_signals = SMCAnalyzer.detect_bos_choch(df)
                fvg = SMCAnalyzer.find_fvg(df)
                sr_levels = SMCAnalyzer.find_support_resistance(df)
                trend = SMCAnalyzer.analyze_trend(df)
                
                tf_data[tf] = {
                    'dataframe': df,
                    'current_price': float(df['close'].iloc[-1]),
                    'high_24h': float(df['high'].tail(24).max() if tf == '1h' else df['high'].tail(7).max()),
                    'low_24h': float(df['low'].tail(24).min() if tf == '1h' else df['low'].tail(7).min()),
                    'volume_trend': 'Increasing' if df['volume'].tail(10).mean() > df['volume'].tail(30).mean() else 'Decreasing',
                    'trend': trend,
                    'order_blocks': order_blocks,
                    'bos_signals': bos_signals,
                    'fvg': fvg,
                    'support': sr_levels['support'],
                    'resistance': sr_levels['resistance']
                }
                
                print(f"  ✓ {tf}: {len(df)} candles | Trend: {trend}")
                await asyncio.sleep(0.3)  # Rate limiting
            
            # Get DeepSeek multi-timeframe analysis
            print(f"🤖 Getting AI analysis for {coin}...")
            analysis = await DeepSeekAnalyzer.analyze_multi_timeframe(session, coin, tf_data)
            
            if analysis and analysis['signal'] in ['BUY', 'SELL']:
                print(f"  🎯 Signal: {analysis['signal']} | Confidence: {analysis['confidence']}% | Alignment: {analysis['timeframe_alignment']}")
                
                # Generate chart
                chart_buffer = ChartGenerator.create_mtf_chart(coin, tf_data, analysis)
                
                # Send alert
                await self.send_telegram_alert(coin, analysis, chart_buffer)
            else:
                print(f"  ⏸️ {coin}: No actionable signal (WAIT)")
            
        except Exception as e:
            print(f"❌ Error analyzing {coin}: {str(e)}")
            traceback.print_exc()
    
    async def scan_all_coins(self):
        """Scan all coins with multi-timeframe analysis"""
        connector = aiohttp.TCPConnector(limit=30, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=60, connect=15, sock_read=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for coin in COINS:
                await self.analyze_coin_mtf(session, coin)
                await asyncio.sleep(2)  # Spacing between coins
    
    async def run(self):
        """Main bot loop"""
        self.is_running = True
        print("🚀 Multi-Timeframe Trading Bot Started!")
        print(f"📊 Coins: {', '.join(COINS)}")
        print(f"⏱️ Timeframes: {' → '.join(TIMEFRAMES)} (HTF to LTF)")
        print(f"🔄 Scan Interval: {SCAN_INTERVAL}s\n")
        
        while self.is_running:
            try:
                print(f"\n{'='*60}")
                print(f"🔍 Multi-Timeframe Scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                
                await self.scan_all_coins()
                
                print(f"\n✅ Scan completed. Next scan in {SCAN_INTERVAL}s ({SCAN_INTERVAL//60} minutes)")
                await asyncio.sleep(SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                self.is_running = False
                break
            except Exception as e:
                print(f"\n❌ Critical error: {str(e)}")
                traceback.print_exc()
                await asyncio.sleep(60)

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.run())
