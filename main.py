import os
import asyncio
import aiohttp
from datetime import datetime
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
TIMEFRAMES = ['4h', '1h']  # 4H for trend/setup, 1H for entry timing
SCAN_INTERVAL = 3600  # 1 hour

class BinanceAPI:
    """Binance Futures API"""
    BASE_URL = "https://fapi.binance.com/fapi/v1"
    
    @staticmethod
    async def get_candlestick_data(session, symbol, timeframe, limit=1000):
        """Fetch candlestick data from Binance Futures"""
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
    """Smart Money Concepts Analyzer"""
    
    @staticmethod
    def find_order_blocks(df, lookback=20):
        """Identify order blocks - institutional buy/sell zones"""
        order_blocks = {'bullish': [], 'bearish': []}
        
        for i in range(lookback, len(df) - 1):
            # Bullish Order Block (strong buying candle)
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
            
            # Bearish Order Block (strong selling candle)
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
        """Detect Break of Structure (BOS) and Change of Character (ChoCH)"""
        signals = []
        highs = df['high'].rolling(window=swing_period).max()
        lows = df['low'].rolling(window=swing_period).min()
        
        for i in range(swing_period, len(df) - 1):
            # Bullish BOS - break above previous swing high
            if df['close'].iloc[i] > highs.iloc[i-1]:
                signals.append({
                    'type': 'BOS_BULL',
                    'price': float(df['close'].iloc[i]),
                    'timestamp': str(df['timestamp'].iloc[i])
                })
            
            # Bearish BOS - break below previous swing low
            if df['close'].iloc[i] < lows.iloc[i-1]:
                signals.append({
                    'type': 'BOS_BEAR',
                    'price': float(df['close'].iloc[i]),
                    'timestamp': str(df['timestamp'].iloc[i])
                })
        
        return signals
    
    @staticmethod
    def find_fvg(df):
        """Find Fair Value Gaps (imbalances in price)"""
        fvgs = {'bullish': [], 'bearish': []}
        
        for i in range(2, len(df)):
            # Bullish FVG - gap up indicating strong buying
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                fvgs['bullish'].append({
                    'top': float(df['low'].iloc[i]),
                    'bottom': float(df['high'].iloc[i-2]),
                    'size': float(gap_size),
                    'timestamp': str(df['timestamp'].iloc[i])
                })
            
            # Bearish FVG - gap down indicating strong selling
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                fvgs['bearish'].append({
                    'top': float(df['low'].iloc[i-2]),
                    'bottom': float(df['high'].iloc[i]),
                    'size': float(gap_size),
                    'timestamp': str(df['timestamp'].iloc[i])
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
        """Determine trend using moving averages"""
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        current_price = df['close'].iloc[-1]
        ema20 = df['ema20'].iloc[-1]
        ema50 = df['ema50'].iloc[-1]
        
        # Strong uptrend
        if current_price > ema20 > ema50:
            return 'STRONG_BULLISH'
        # Moderate uptrend
        elif current_price > ema20 and ema20 < ema50:
            return 'BULLISH'
        # Strong downtrend
        elif current_price < ema20 < ema50:
            return 'STRONG_BEARISH'
        # Moderate downtrend
        elif current_price < ema20 and ema20 > ema50:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    @staticmethod
    def detect_candlestick_patterns(df):
        """Detect key candlestick patterns"""
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
            
            # Hammer (bullish reversal)
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            lower_shadow = min(df['close'].iloc[i], df['open'].iloc[i]) - df['low'].iloc[i]
            upper_shadow = df['high'].iloc[i] - max(df['close'].iloc[i], df['open'].iloc[i])
            
            if lower_shadow > body * 2 and upper_shadow < body * 0.3:
                patterns.append({'type': 'Hammer', 'index': i})
            
            # Shooting Star (bearish reversal)
            if upper_shadow > body * 2 and lower_shadow < body * 0.3:
                patterns.append({'type': 'Shooting_Star', 'index': i})
        
        return patterns

class DeepSeekAnalyzer:
    """DeepSeek V3 AI for 4H + 1H Combined Analysis"""
    
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    @staticmethod
    async def analyze_4h_1h_strategy(session, coin, tf_data):
        """
        4H + 1H Strategy:
        - 4H: Trend direction, entry zones, key S/R levels
        - 1H: Precise entry timing, confirmation signals
        """
        
        prompt = f"""You are a professional crypto trader using 4H + 1H timeframe strategy with Smart Money Concepts.

**TRADING STRATEGY:**
- **4H Chart (Higher TF):** Identifies overall trend, entry zones, key support/resistance, and order blocks
- **1H Chart (Lower TF):** Provides precise entry timing, confirms 4H setup, fine-tunes targets

**{coin}/USDT ANALYSIS:**

📊 **4-HOUR TIMEFRAME (Trend & Setup):**
- Current Price: ${tf_data['4h']['current_price']:,.2f}
- Trend: {tf_data['4h']['trend']}
- Support Levels: {tf_data['4h']['support']}
- Resistance Levels: {tf_data['4h']['resistance']}
- Bullish Order Blocks: {len(tf_data['4h']['order_blocks']['bullish'])} zones
- Bearish Order Blocks: {len(tf_data['4h']['order_blocks']['bearish'])} zones
- Bullish FVG: {len(tf_data['4h']['fvg']['bullish'])}
- Bearish FVG: {len(tf_data['4h']['fvg']['bearish'])}
- BOS Signals: {len([s for s in tf_data['4h']['bos_signals'] if 'BULL' in s['type']])} bullish, {len([s for s in tf_data['4h']['bos_signals'] if 'BEAR' in s['type']])} bearish
- Recent Patterns: {[p['type'] for p in tf_data['4h']['patterns'][-3:]]}

⏱️ **1-HOUR TIMEFRAME (Entry Timing):**
- Current Price: ${tf_data['1h']['current_price']:,.2f}
- Trend: {tf_data['1h']['trend']}
- 24H High: ${tf_data['1h']['high_24h']:,.2f}
- 24H Low: ${tf_data['1h']['low_24h']:,.2f}
- Support Levels: {tf_data['1h']['support'][:3]}
- Resistance Levels: {tf_data['1h']['resistance'][:3]}
- Volume Trend: {tf_data['1h']['volume_trend']}
- Recent Patterns: {[p['type'] for p in tf_data['1h']['patterns'][-3:]]}
- Order Blocks: Bull={len(tf_data['1h']['order_blocks']['bullish'])}, Bear={len(tf_data['1h']['order_blocks']['bearish'])}

**SIGNAL GENERATION RULES:**

✅ **BUY Signal Requirements:**
1. 4H must show BULLISH or STRONG_BULLISH trend
2. 4H must have bullish order block or FVG near current price
3. 1H must confirm with bullish pattern or break above resistance
4. 1H volume must be increasing on bullish candles
5. Risk:Reward must be at least 1:2

✅ **SELL Signal Requirements:**
1. 4H must show BEARISH or STRONG_BEARISH trend
2. 4H must have bearish order block or FVG near current price
3. 1H must confirm with bearish pattern or break below support
4. 1H volume must be increasing on bearish candles
5. Risk:Reward must be at least 1:2

⏸️ **WAIT Signal:**
- If 4H and 1H trends conflict
- If no clear setup on 4H
- If 1H doesn't confirm 4H bias
- If risk:reward is poor (<1:2)

**OUTPUT FORMAT (JSON):**
{{
    "signal": "BUY/SELL/WAIT",
    "confidence": 0-100,
    "4h_bias": "BULLISH/BEARISH/NEUTRAL",
    "1h_confirmation": "CONFIRMED/PENDING/REJECTED",
    "timeframe_sync": "ALIGNED/PARTIAL/CONFLICT",
    "entry_strategy": "aggressive/conservative",
    "entry_range": [lower, upper],
    "stop_loss": price,
    "take_profits": [tp1, tp2, tp3],
    "risk_reward": "1:X",
    "key_levels": {{
        "4h_support": [s1, s2],
        "4h_resistance": [r1, r2],
        "1h_entry_zone": [lower, upper]
    }},
    "reasoning": "Brief 2-3 sentence explanation of why 4H setup + 1H timing creates this signal"
}}

**IMPORTANT:** Be conservative. Only give BUY/SELL when 4H clearly defines the setup and 1H confirms entry timing."""

        headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': 'You are an expert trader specializing in 4H + 1H timeframe strategy with Smart Money Concepts.'},
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
    """Generate 4H + 1H dual timeframe charts"""
    
    @staticmethod
    def create_dual_tf_chart(coin, tf_data, analysis):
        """Create 2-panel chart: 4H top, 1H bottom with clear trade levels"""
        
        fig = plt.figure(figsize=(18, 11), facecolor='#0a0e27')
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.25)
        
        timeframes = ['4h', '1h']
        titles = [
            '4-HOUR CHART - Trend & Setup Zone',
            '1-HOUR CHART - Entry Timing'
        ]
        
        axes = []
        for idx, (tf, title) in enumerate(zip(timeframes, titles)):
            ax = fig.add_subplot(gs[idx])
            ax.set_facecolor('#0f1429')
            df = tf_data[tf]['dataframe'].tail(100).reset_index(drop=True)
            
            # Plot candlesticks
            for i in range(len(df)):
                o, h, l, c = df['open'].iloc[i], df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i]
                color = '#00ff88' if c >= o else '#ff3366'
                
                body_height = abs(c - o)
                body_bottom = min(o, c)
                
                # Candle body
                ax.add_patch(patches.Rectangle(
                    (i - 0.4, body_bottom), 0.8, body_height,
                    facecolor=color, edgecolor=color, linewidth=1, alpha=0.9
                ))
                
                # Wicks
                ax.plot([i, i], [l, h], color=color, linewidth=1.2, alpha=0.7)
            
            # Support/Resistance
            if tf_data[tf]['support']:
                for level in tf_data[tf]['support'][-2:]:
                    ax.axhline(y=level, color='#00ff88', linestyle='--', 
                              alpha=0.4, linewidth=1.5)
            
            if tf_data[tf]['resistance']:
                for level in tf_data[tf]['resistance'][-2:]:
                    ax.axhline(y=level, color='#ff3366', linestyle='--', 
                              alpha=0.4, linewidth=1.5)
            
            # Order blocks on 4H
            if tf == '4h':
                obs = tf_data[tf]['order_blocks']
                if obs['bullish']:
                    latest_bull_ob = obs['bullish'][-1]
                    ax.axhspan(latest_bull_ob['price'], latest_bull_ob['high'],
                              alpha=0.12, color='#00ff88')
                if obs['bearish']:
                    latest_bear_ob = obs['bearish'][-1]
                    ax.axhspan(latest_bear_ob['low'], latest_bear_ob['price'],
                              alpha=0.12, color='#ff3366')
            
            ax.set_title(f'{title} | Trend: {tf_data[tf]["trend"]}', 
                        fontsize=13, fontweight='bold', color='white', pad=12)
            ax.set_ylabel('Price (USDT)', fontsize=11, color='white')
            ax.tick_params(colors='white', labelsize=9)
            ax.grid(True, alpha=0.15, linestyle=':', color='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')
            axes.append(ax)
        
        # MARK ENTRY, SL, TPs on BOTH charts
        if analysis and analysis['signal'] in ['BUY', 'SELL']:
            signal_color = '#00ff88' if analysis['signal'] == 'BUY' else '#ff3366'
            entry_lower, entry_upper = analysis['entry_range']
            stop_loss = analysis['stop_loss']
            tp1, tp2, tp3 = analysis['take_profits']
            
            for ax_idx, ax in enumerate(axes):
                # Entry Zone (shaded box)
                ax.axhspan(entry_lower, entry_upper, 
                          alpha=0.25, color=signal_color, zorder=1)
                ax.axhline(y=entry_lower, color=signal_color, linestyle='-', 
                          linewidth=2.5, alpha=0.9, label='Entry Zone')
                ax.axhline(y=entry_upper, color=signal_color, linestyle='-', 
                          linewidth=2.5, alpha=0.9)
                
                # Add ENTRY text
                ax.text(len(tf_data[timeframes[ax_idx]]['dataframe'].tail(100)) - 5, 
                       (entry_lower + entry_upper) / 2, 
                       f'  ENTRY\n  ${entry_lower:,.0f}-${entry_upper:,.0f}',
                       color=signal_color, fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                                edgecolor=signal_color, linewidth=2, alpha=0.8),
                       verticalalignment='center')
                
                # Stop Loss (red line)
                ax.axhline(y=stop_loss, color='#ff0000', linestyle='--', 
                          linewidth=2.5, alpha=0.9, label='Stop Loss')
                ax.text(len(tf_data[timeframes[ax_idx]]['dataframe'].tail(100)) - 5, 
                       stop_loss, 
                       f'  STOP LOSS: ${stop_loss:,.0f}',
                       color='#ff0000', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='black', 
                                edgecolor='#ff0000', linewidth=2, alpha=0.8),
                       verticalalignment='center')
                
                # Take Profits
                for tp_num, tp_val in enumerate([tp1, tp2, tp3], 1):
                    ax.axhline(y=tp_val, color='#ffdd00', linestyle='-.', 
                              linewidth=2, alpha=0.85)
                    ax.text(len(tf_data[timeframes[ax_idx]]['dataframe'].tail(100)) - 5, 
                           tp_val, 
                           f'  TP{tp_num}: ${tp_val:,.0f}',
                           color='#ffdd00', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                                    edgecolor='#ffdd00', linewidth=1.5, alpha=0.8),
                           verticalalignment='center')
            
            # Signal Info Box
            signal_text = f"""
╔═══════════════════════╗
║  {analysis['signal']} SIGNAL  ║
╚═══════════════════════╝

Confidence: {analysis['confidence']}%
4H Bias: {analysis['4h_bias']}
1H Confirm: {analysis['1h_confirmation']}
TF Sync: {analysis['timeframe_sync']}

ENTRY: ${entry_lower:,.0f} - ${entry_upper:,.0f}
STOP LOSS: ${stop_loss:,.0f}

TARGETS:
  TP1: ${tp1:,.0f}
  TP2: ${tp2:,.0f}
  TP3: ${tp3:,.0f}

Risk:Reward: {analysis['risk_reward']}
"""
            fig.text(0.985, 0.5, signal_text,
                    transform=fig.transFigure,
                    fontsize=9.5, fontweight='bold',
                    family='monospace',
                    color=signal_color,
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#0a0e27', 
                             edgecolor=signal_color, linewidth=2.5, alpha=0.95),
                    verticalalignment='center',
                    horizontalalignment='right')
        
        plt.suptitle(f'{coin}/USDT - 4H + 1H Strategy Analysis', 
                    fontsize=17, fontweight='bold', color='white', y=0.985)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        fig.text(0.99, 0.01, f'Generated: {timestamp}', 
                ha='right', fontsize=8, color='gray')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=160, facecolor='#0a0e27', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf

class TradingBot:
    """Main Bot with 4H + 1H Strategy"""
    
    def __init__(self):
        request = HTTPXRequest(
            connection_pool_size=20,
            connect_timeout=30,
            read_timeout=30,
            write_timeout=30,
            pool_timeout=30
        )
        self.telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN, request=request)
        self.is_running = False
        self.alert_semaphore = asyncio.Semaphore(3)
    
    async def send_telegram_alert(self, coin, analysis, chart_buffer):
        """Send alert with rate limiting"""
        async with self.alert_semaphore:
            try:
                if analysis['signal'] == 'WAIT':
                    return
                
                signal_emoji = "🟢" if analysis['signal'] == 'BUY' else "🔴"
                
                message = f"""
{signal_emoji} **{analysis['signal']} SIGNAL** {signal_emoji}

**{coin}/USDT**
**Confidence:** {analysis['confidence']}%
**Strategy:** 4H + 1H Combined

📊 **Timeframe Analysis:**
• 4H Bias: {analysis['4h_bias']}
• 1H Confirmation: {analysis['1h_confirmation']}
• Sync Status: {analysis['timeframe_sync']}
• Entry Style: {analysis['entry_strategy'].title()}

💰 **Trade Setup:**
**Entry Zone:** ${analysis['entry_range'][0]:,.2f} - ${analysis['entry_range'][1]:,.2f}
**Stop Loss:** ${analysis['stop_loss']:,.2f}
**Targets:**
  TP1: ${analysis['take_profits'][0]:,.2f}
  TP2: ${analysis['take_profits'][1]:,.2f}
  TP3: ${analysis['take_profits'][2]:,.2f}

**Risk:Reward:** {analysis['risk_reward']}

📈 **Key Levels:**
4H Support: {', '.join([f'${s:,.0f}' for s in analysis['key_levels']['4h_support'][:2]])}
4H Resistance: {', '.join([f'${r:,.0f}' for r in analysis['key_levels']['4h_resistance'][:2]])}
1H Entry Zone: ${analysis['key_levels']['1h_entry_zone'][0]:,.0f} - ${analysis['key_levels']['1h_entry_zone'][1]:,.0f}

💡 **Analysis:**
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
                
                print(f"✅ {coin}: {analysis['signal']} ({analysis['confidence']}%) | 4H: {analysis['4h_bias']} | 1H: {analysis['1h_confirmation']}")
                await asyncio.sleep(1)
                
            except TelegramError as e:
                print(f"⚠️ Telegram error for {coin}: {str(e)}")
            except Exception as e:
                print(f"❌ Error sending alert for {coin}: {str(e)}")
    
    async def analyze_coin_4h_1h(self, session, coin):
        """Analyze coin using 4H + 1H strategy"""
        try:
            print(f"\n📊 {coin} | Fetching 4H + 1H data...")
            
            tf_data = {}
            
            # Fetch both timeframes
            for tf in TIMEFRAMES:
                df = await BinanceAPI.get_candlestick_data(session, coin, tf, limit=1000)
                
                if df is None or len(df) < 100:
                    print(f"  ⚠️ Insufficient data for {tf}")
                    return
                
                #
