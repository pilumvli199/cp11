import os
import asyncio
import logging
from datetime import datetime, timedelta
import json
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes
import requests
from openai import OpenAI
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import io
from PIL import Image

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

client = OpenAI(api_key=OPENAI_API_KEY)

oi_storage = {}

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
SYMBOLS = ['BTC-PERPETUAL', 'ETH-PERPETUAL']
TIMEFRAMES = ['30', '60', '240']
MAX_TRADES_PER_DAY = 8
CANDLE_COUNT = 500

class DeribitClient:
    
    RESOLUTION_MAP = {
        '30': '30',
        '60': '60',
        '240': '1D'
    }
    
    @staticmethod
    def get_candles(symbol: str, timeframe: str, count: int = CANDLE_COUNT) -> pd.DataFrame:
        
        logger.info(f"📊 Fetching {count} candles for {symbol} {timeframe}m...")
        
        resolution = DeribitClient.RESOLUTION_MAP.get(timeframe, timeframe)
        url = f"{DERIBIT_BASE}/get_tradingview_chart_data"
        
        tf_minutes = int(timeframe)
        if timeframe == '240':
            days_needed = count + 10
        else:
            days_needed = (count * tf_minutes) // (24 * 60) + 10
        
        params = {
            'instrument_name': symbol,
            'resolution': resolution,
            'start_timestamp': int((datetime.now() - timedelta(days=days_needed)).timestamp() * 1000),
            'end_timestamp': int(datetime.now().timestamp() * 1000)
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"❌ Deribit API HTTP {response.status_code} for {symbol} {timeframe}m")
                return pd.DataFrame()
            
            data = response.json()
            
            if 'error' in data:
                logger.error(f"❌ Deribit API error for {symbol} {timeframe}m: {data['error']}")
                return pd.DataFrame()
            
            if 'result' not in data:
                logger.error(f"❌ No result in Deribit response for {symbol} {timeframe}m")
                return pd.DataFrame()
            
            result = data['result']
            
            if result.get('status') != 'ok':
                logger.error(f"❌ Deribit status not OK for {symbol} {timeframe}m")
                return pd.DataFrame()
            
            required_fields = ['ticks', 'open', 'high', 'low', 'close', 'volume']
            missing = [f for f in required_fields if f not in result]
            if missing:
                logger.error(f"❌ Missing fields for {symbol} {timeframe}m: {missing}")
                return pd.DataFrame()
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(result['ticks'], unit='ms'),
                'open': result['open'],
                'high': result['high'],
                'low': result['low'],
                'close': result['close'],
                'volume': result['volume']
            })
            
            if len(df) == 0:
                logger.warning(f"⚠️ Empty dataframe for {symbol} {timeframe}m")
                return pd.DataFrame()
            
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"✅ Fetched {len(df)} candles for {symbol} {resolution}")
            return df.tail(count)
            
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Timeout fetching {symbol} {timeframe}m")
        except Exception as e:
            logger.error(f"💥 Error fetching {symbol} {timeframe}m: {e}")
        
        return pd.DataFrame()
    
    @staticmethod
    def get_order_book(symbol: str, depth: int = 10) -> Dict:
        logger.info(f"📖 Fetching order book for {symbol}...")
        
        url = f"{DERIBIT_BASE}/get_order_book"
        params = {'instrument_name': symbol, 'depth': depth}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'result' in data:
                oi = data['result'].get('open_interest', 0)
                volume = data['result'].get('stats', {}).get('volume', 0)
                mark = data['result'].get('mark_price', 0)
                
                logger.info(f"📊 {symbol} OI: {oi:,.0f}, Vol: {volume:,.2f}, Mark: ${mark:,.2f}")
                
                return {
                    'open_interest': oi,
                    'volume_24h': volume,
                    'mark_price': mark
                }
        except Exception as e:
            logger.error(f"❌ Error fetching order book for {symbol}: {e}")
        
        return {'open_interest': 0, 'volume_24h': 0, 'mark_price': 0}

class TechnicalAnalyzer:
    
    @staticmethod
    def find_swing_points(df: pd.DataFrame, period: int = 5) -> Tuple[List, List]:
        swing_highs = []
        swing_lows = []
        
        for i in range(period, len(df) - period):
            if df['high'].iloc[i] == df['high'].iloc[i-period:i+period+1].max():
                swing_highs.append({
                    'price': df['high'].iloc[i], 
                    'index': i,
                    'timestamp': df.index[i]
                })
            
            if df['low'].iloc[i] == df['low'].iloc[i-period:i+period+1].min():
                swing_lows.append({
                    'price': df['low'].iloc[i], 
                    'index': i,
                    'timestamp': df.index[i]
                })
        
        current_price = df['close'].iloc[-1]
        
        relevant_highs = [
            s for s in swing_highs 
            if s['index'] >= len(df) - 50 and
               s['price'] >= current_price and
               s['price'] <= current_price * 1.05
        ]
        
        relevant_lows = [
            s for s in swing_lows 
            if s['index'] >= len(df) - 50 and
               s['price'] <= current_price and
               s['price'] >= current_price * 0.95
        ]
        
        if len(relevant_highs) == 0:
            relevant_highs = sorted(swing_highs, key=lambda x: abs(x['price'] - current_price))[:3]
        
        if len(relevant_lows) == 0:
            relevant_lows = sorted(swing_lows, key=lambda x: abs(x['price'] - current_price))[:3]
        
        relevant_highs = sorted(relevant_highs, key=lambda x: x['index'], reverse=True)[:3]
        relevant_lows = sorted(relevant_lows, key=lambda x: x['index'], reverse=True)[:3]
        
        logger.info(f"🎯 Found {len(relevant_highs)} resistance & {len(relevant_lows)} support levels")
        
        return relevant_highs, relevant_lows
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> List[Dict]:
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = abs(last['close'] - last['open'])
        upper_wick = last['high'] - max(last['open'], last['close'])
        lower_wick = min(last['open'], last['close']) - last['low']
        candle_range = last['high'] - last['low']
        
        if candle_range > 0:
            if body < candle_range * 0.1:
                patterns.append({'type': 'candlestick', 'name': 'Doji', 'signal': 'neutral'})
            
            if lower_wick > body * 2 and upper_wick < body * 0.5 and last['close'] > last['open']:
                patterns.append({'type': 'candlestick', 'name': 'Hammer', 'signal': 'bullish'})
            
            if upper_wick > body * 2 and lower_wick < body * 0.5 and last['close'] < last['open']:
                patterns.append({'type': 'candlestick', 'name': 'Shooting Star', 'signal': 'bearish'})
            
            if (prev['close'] < prev['open'] and 
                last['close'] > last['open'] and 
                last['close'] > prev['open'] and 
                last['open'] < prev['close']):
                patterns.append({'type': 'candlestick', 'name': 'Bullish Engulfing', 'signal': 'bullish'})
            
            if (prev['close'] > prev['open'] and 
                last['close'] < last['open'] and 
                last['close'] < prev['open'] and 
                last['open'] > prev['close']):
                patterns.append({'type': 'candlestick', 'name': 'Bearish Engulfing', 'signal': 'bearish'})
        
        if len(df) >= 20:
            recent = df.tail(20)
            highs = recent['high'].values
            lows = recent['low'].values
            
            if max(highs[-10:]) < max(highs[:10]) and min(lows[-10:]) > min(lows[:10]):
                patterns.append({'type': 'chart', 'name': 'Symmetrical Triangle', 'signal': 'breakout_pending'})
            elif abs(max(highs[-10:]) - max(highs[:10])) < (max(highs) * 0.01) and min(lows[-10:]) > min(lows[:10]):
                patterns.append({'type': 'chart', 'name': 'Ascending Triangle', 'signal': 'bullish'})
            elif abs(min(lows[-10:]) - min(lows[:10])) < (min(lows) * 0.01) and max(highs[-10:]) < max(highs[:10]):
                patterns.append({'type': 'chart', 'name': 'Descending Triangle', 'signal': 'bearish'})
        
        logger.info(f"🔍 Detected {len(patterns)} patterns: {[p['name'] for p in patterns]}")
        return patterns
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame) -> Dict:
        price_range = df['high'].max() - df['low'].min()
        bins = 20
        bin_size = price_range / bins
        
        volume_by_price = {}
        for _, row in df.iterrows():
            price_bin = int((row['close'] - df['low'].min()) / bin_size)
            volume_by_price[price_bin] = volume_by_price.get(price_bin, 0) + row['volume']
        
        if volume_by_price:
            poc_bin = max(volume_by_price, key=volume_by_price.get)
            poc_price = df['low'].min() + (poc_bin * bin_size)
            
            return {
                'poc': poc_price,
                'high_volume_nodes': sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)[:3]
            }
        return {'poc': None, 'high_volume_nodes': []}

class OITracker:
    
    @staticmethod
    def store_oi(symbol: str, oi_data: Dict):
        timestamp = int(datetime.now().timestamp())
        
        if symbol not in oi_storage:
            oi_storage[symbol] = []
        
        oi_storage[symbol].append({
            'timestamp': timestamp,
            'data': oi_data
        })
        
        cutoff = timestamp - (3 * 3600)
        oi_storage[symbol] = [
            entry for entry in oi_storage[symbol] 
            if entry['timestamp'] >= cutoff
        ]
        
        logger.info(f"💾 Stored OI data for {symbol}: OI={oi_data.get('open_interest', 0):,.0f}")
    
    @staticmethod
    def get_oi_history(symbol: str, hours: int = 2) -> List[Dict]:
        cutoff = int((datetime.now() - timedelta(hours=hours)).timestamp())
        
        if symbol not in oi_storage:
            return []
        
        history = [
            {**entry['data'], 'timestamp': entry['timestamp']}
            for entry in oi_storage[symbol]
            if entry['timestamp'] >= cutoff
        ]
        
        logger.info(f"📚 Found {len(history)} OI records for {symbol}")
        return sorted(history, key=lambda x: x['timestamp'])
    
    @staticmethod
    def analyze_oi_trend(symbol: str) -> Dict:
        history = OITracker.get_oi_history(symbol, hours=2)
        
        if len(history) < 2:
            logger.warning(f"⚠️ Insufficient OI data for {symbol}")
            return {'trend': 'insufficient_data', 'change': 0, 'supporting_sr': None}
        
        current_oi = history[-1]['open_interest']
        old_oi = history[0]['open_interest']
        change = ((current_oi - old_oi) / old_oi * 100) if old_oi > 0 else 0
        
        if change > 5:
            trend = 'strongly_increasing'
        elif change > 2:
            trend = 'increasing'
        elif change < -5:
            trend = 'strongly_decreasing'
        elif change < -2:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        logger.info(f"📊 {symbol} OI Trend: {trend} ({change:+.2f}%)")
        
        return {
            'trend': trend,
            'change': round(change, 2),
            'current_oi': current_oi,
            'previous_oi': old_oi,
            'supporting_sr': None
        }

class ChartGenerator:
    
    @staticmethod
    def create_chart(df: pd.DataFrame, analysis: Dict, ai_result: Dict, symbol: str) -> io.BytesIO:
        
        logger.info(f"📈 Generating chart for {symbol} based on GPT analysis...")
        
        mc = mpf.make_marketcolors(
            up='#26a69a',
            down='#ef5350',
            edge='inherit',
            wick='inherit',
            volume='in',
            alpha=0.9
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='-',
            gridcolor='#e0e0e0',
            facecolor='white',
            figcolor='white',
            y_on_right=False
        )
        
        fig, axes = mpf.plot(
            df.tail(100),
            type='candle',
            style=s,
            volume=True,
            returnfig=True,
            figsize=(14, 8),
            title=f"\n{symbol} - {ai_result.get('pattern', 'Analysis')}",
            ylabel='Price ($)',
            ylabel_lower='Volume'
        )
        
        ax = axes[0]
        
        if 'swing_lows_30m' in analysis and analysis['swing_lows_30m']:
            for swing in analysis['swing_lows_30m'][-3:]:
                price = swing['price']
                ax.axhline(y=price, color='#4caf50', linestyle='--', linewidth=1.5, alpha=0.7, label='Support 30m')
        
        if 'swing_highs_30m' in analysis and analysis['swing_highs_30m']:
            for swing in analysis['swing_highs_30m'][-3:]:
                price = swing['price']
                ax.axhline(y=price, color='#f44336', linestyle='--', linewidth=1.5, alpha=0.7, label='Resistance 30m')
        
        if analysis.get('support_4h'):
            ax.axhline(y=analysis['support_4h'], color='#2e7d32', linestyle='-', linewidth=2, label='Support 4H')
        
        if analysis.get('resistance_4h'):
            ax.axhline(y=analysis['resistance_4h'], color='#c62828', linestyle='-', linewidth=2, label='Resistance 4H')
        
        current_price = df['close'].iloc[-1]
        ax.axhline(y=current_price, color='#ff9800', linestyle=':', linewidth=2, label=f'Current: ${current_price:.2f}')
        
        if ai_result.get('entry') and ai_result['signal'] in ['LONG', 'SHORT']:
            entry_price = ai_result['entry']
            ax.axhline(y=entry_price, color='#2196f3', linestyle='-.', linewidth=2.5, label=f'Entry: ${entry_price:.2f}')
            
            if ai_result.get('sl'):
                sl_price = ai_result['sl']
                ax.axhline(y=sl_price, color='#d32f2f', linestyle='-.', linewidth=2, label=f'Stop Loss: ${sl_price:.2f}')
            
            if ai_result.get('target'):
                target_price = ai_result['target']
                ax.axhline(y=target_price, color='#388e3c', linestyle='-.', linewidth=2, label=f'Target: ${target_price:.2f}')
        
        if analysis.get('patterns'):
            pattern_text = "\n".join([p['name'] for p in analysis['patterns'][:3]])
            ax.text(
                0.02, 0.98, 
                f"Patterns:\n{pattern_text}",
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10
            )
        
        if ai_result.get('signal'):
            signal = ai_result['signal']
            signal_color = '#4caf50' if 'LONG' in signal else '#f44336' if 'SHORT' in signal else '#9e9e9e'
            
            signal_text = f"GPT SIGNAL: {signal}\n"
            if ai_result.get('entry'):
                signal_text += f"Entry: ${ai_result['entry']:.2f}\n"
            if ai_result.get('sl'):
                signal_text += f"SL: ${ai_result['sl']:.2f}\n"
            if ai_result.get('target'):
                signal_text += f"Target: ${ai_result['target']:.2f}\n"
            if ai_result.get('reason'):
                reason_short = ai_result['reason'][:60] + "..." if len(ai_result['reason']) > 60 else ai_result['reason']
                signal_text += f"\n{reason_short}"
            
            ax.text(
                0.98, 0.98,
                signal_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=signal_color, alpha=0.3, edgecolor=signal_color, linewidth=2),
                fontsize=10,
                fontweight='bold'
            )
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
        
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=150, facecolor='white')
        buf.seek(0)
        plt.close(fig)
        
        logger.info(f"✅ Chart generated for {symbol} with GPT overlay")
        return buf

class TradeAnalyzer:
    
    @staticmethod
    def analyze_setup(symbol: str) -> Dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 ANALYZING {symbol}")
        logger.info(f"{'='*60}")
        
        df_30m = DeribitClient.get_candles(symbol, '30', CANDLE_COUNT)
        df_1h = DeribitClient.get_candles(symbol, '60', CANDLE_COUNT)
        df_4h = DeribitClient.get_candles(symbol, '240', CANDLE_COUNT)
        
        if df_30m.empty or df_1h.empty or df_4h.empty:
            logger.error(f"❌ Insufficient data for {symbol}")
            return {'valid': False, 'reason': 'Insufficient data', 'symbol': symbol}
        
        logger.info(f"✅ {symbol}: Loaded {len(df_30m)}x30m, {len(df_1h)}x1h, {len(df_4h)}x4h candles")
        
        oi_data = DeribitClient.get_order_book(symbol)
        OITracker.store_oi(symbol, oi_data)
        oi_trend = OITracker.analyze_oi_trend(symbol)
        
        swing_highs_30m, swing_lows_30m = TechnicalAnalyzer.find_swing_points(df_30m)
        swing_highs_1h, swing_lows_1h = TechnicalAnalyzer.find_swing_points(df_1h)
        swing_highs_4h, swing_lows_4h = TechnicalAnalyzer.find_swing_points(df_4h)
        
        logger.info(f"📍 30m: {len(swing_highs_30m)} resistance, {len(swing_lows_30m)} support")
        logger.info(f"📍 1h: {len(swing_highs_1h)} resistance, {len(swing_lows_1h)} support")
        logger.info(f"📍 4h: {len(swing_highs_4h)} resistance, {len(swing_lows_4h)} support")
        
        patterns = TechnicalAnalyzer.detect_patterns(df_30m)
        volume_profile = TechnicalAnalyzer.calculate_volume_profile(df_30m)
        
        current_price = df_30m['close'].iloc[-1]
        avg_volume = df_30m['volume'].tail(20).mean()
        current_volume = df_30m['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        logger.info(f"💰 Price: ${current_price:.2f}")
        logger.info(f"📊 Volume: {volume_ratio:.2f}x")
        
        resistance_4h = None
        support_4h = None
        
        if swing_highs_4h:
            resistances = [s['price'] for s in swing_highs_4h if s['price'] >= current_price]
            resistance_4h = min(resistances) if resistances else swing_highs_4h[0]['price']
        
        if swing_lows_4h:
            supports = [s['price'] for s in swing_lows_4h if s['price'] <= current_price]
            support_4h = max(supports) if supports else swing_lows_4h[0]['price']
        
        logger.info(f"🎯 4H Resistance: ${resistance_4h:.2f if resistance_4h else 0}")
        logger.info(f"🎯 4H Support: ${support_4h:.2f if support_4h else 0}")
        
        analysis = {
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'patterns': patterns,
            'volume_ratio': round(volume_ratio, 2),
            'oi_trend': oi_trend,
            'swing_highs_30m': swing_highs_30m,
            'swing_lows_30m': swing_lows_30m,
            'swing_highs_1h': swing_highs_1h,
            'swing_lows_1h': swing_lows_1h,
            'resistance_4h': resistance_4h,
            'support_4h': support_4h,
            'volume_profile': volume_profile,
            'df_30m': df_30m,
            'valid': True
        }
        
        logger.info(f"✅ Analysis complete for {symbol}")
        return analysis
    
    @staticmethod
    def get_ai_analysis(analysis: Dict) -> Dict:
        
        logger.info(f"\n🤖 Calling OpenAI GPT-4o-mini for {analysis['symbol']}...")
        
        patterns_text = "\n".join([f"- {p['name']} ({p['signal']})" for p in analysis.get('patterns', [])]) if analysis.get('patterns') else "None detected"
        
        support_30m = analysis['swing_lows_30m'][-1]['price'] if analysis.get('swing_lows_30m') else None
        resistance_30m = analysis['swing_highs_30m'][-1]['price'] if analysis.get('swing_highs_30m') else None
        
        support_30m_str = f"${support_30m:.2f}" if support_30m else "N/A"
        resistance_30m_str = f"${resistance_30m:.2f}" if resistance_30m else "N/A"
        support_4h_str = f"${analysis.get('support_4h'):.2f}" if analysis.get('support_4h') else "N/A"
        resistance_4h_str = f"${analysis.get('resistance_4h'):.2f}" if analysis.get('resistance_4h') else "N/A"
        
        prompt = f"""Analyze this crypto setup for {analysis['symbol']}:

CURRENT PRICE: ${analysis['current_price']:.2f}

PATTERNS:
{patterns_text}

VOLUME: {analysis.get('volume_ratio', 0):.2f}x average

OPEN INTEREST:
- Trend: {analysis.get('oi_trend', {}).get('trend', 'unknown')}
- Change: {analysis.get('oi_trend', {}).get('change', 0):.1f}%

KEY LEVELS:
- 30m Support: {support_30m_str}
- 30m Resistance: {resistance_30m_str}
- 4H Support: {support_4h_str}
- 4H Resistance: {resistance_4h_str}

TRADING RULES:
1. Volume should be >1.2x (you have {analysis.get('volume_ratio', 0):.2f}x)
2. Must have clear breakout/breakdown
3. Reasonable R:R (min 1:1.5)
4. Price near key S/R level

TASK: Decide if this is tradeable. 

If YES, provide:
SIGNAL: LONG or SHORT
ENTRY: [specific price near current]
SL: [swing level]
TARGET: [with good R:R]
PATTERN: [trigger pattern]
REASON: [why trade]

If NO trade:
SIGNAL: NO_TRADE
REASON: [why not]"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a strict crypto trader. Only take high-probability setups. Be conservative."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            ai_text = response.choices[0].message.content
            logger.info(f"📝 GPT-4o-mini response:\n{ai_text}")
            
            result = {
                'signal': 'NO_TRADE',
                'entry': None,
                'sl': None,
                'target': None,
                'pattern': 'None',
                'reason': 'No clear setup',
                'full_response': ai_text
            }
            
            lines = ai_text.split('\n')
            for line in lines:
                line = line.strip()
                
                if 'SIGNAL:' in line:
                    signal_part = line.split('SIGNAL:')[-1].strip().upper()
                    if 'LONG' in signal_part:
                        result['signal'] = 'LONG'
                    elif 'SHORT' in signal_part:
                        result['signal'] = 'SHORT'
                
                elif 'ENTRY:' in line:
                    try:
                        entry_str = line.split('ENTRY:')[-1].strip()
                        entry_str = entry_str.replace('$', '')
                        entry_str = entry_str.replace(',', '')
                        result['entry'] = float(entry_str.split()[0])
                    except Exception:
                        result['entry'] = analysis['current_price']
                
                elif 'SL:' in line or 'STOP' in line:
                    try:
                        sl_str = line.split(':')[-1].strip()
                        sl_str = sl_str.replace('$', '')
                        sl_str = sl_str.replace(',', '')
                        result['sl'] = float(sl_str.split()[0])
                    except Exception:
                        pass
                
                elif 'TARGET:' in line:
                    try:
                        tgt_str = line.split('TARGET:')[-1].strip()
                        tgt_str = tgt_str.replace(', '')
                        tgt_str = tgt_str.replace(',', '')
                        result['target'] = float(tgt_str.split()[0])
                    except Exception:
                        pass
                
                elif 'PATTERN:' in line:
                    result['pattern'] = line.split('PATTERN:')[-1].strip()
                
                elif 'REASON:' in line:
                    result['reason'] = line.split('REASON:')[-1].strip()
            
            if result['reason'] == 'No clear setup':
                result['reason'] = ai_text
            
            logger.info(f"🎯 Parsed GPT result: {result['signal']} - {result.get('pattern', 'No pattern')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ GPT API error for {analysis['symbol']}: {e}", exc_info=True)
            return {
                'signal': 'ERROR',
                'reason': f'AI analysis failed: {str(e)}',
                'entry': None,
                'sl': None,
                'target': None,
                'pattern': 'Error',
                'full_response': ''
            }

class TradingBot:
    
    def __init__(self):
        self.trade_count_today = 0
        self.last_reset = datetime.now().date()
        self.bot_start_time = datetime.now()
    
    def reset_daily_counter(self):
        if datetime.now().date() > self.last_reset:
            self.trade_count_today = 0
            self.last_reset = datetime.now().date()
            logger.info("🔄 Trade counter reset for new day")
    
    async def scan_markets(self, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 STARTING MARKET SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")
        
        self.reset_daily_counter()
        
        if self.trade_count_today >= MAX_TRADES_PER_DAY:
            logger.info(f"⛔ Daily limit reached: {self.trade_count_today}/{MAX_TRADES_PER_DAY}")
            return
        
        for symbol in SYMBOLS:
            try:
                logger.info(f"\n📊 Scanning {symbol}...")
                analysis = TradeAnalyzer.analyze_setup(symbol)
                
                if not analysis or not analysis.get('valid'):
                    logger.warning(f"⚠️ {symbol}: Invalid data, skipping")
                    continue
                
                logger.info(f"💵 {symbol}: Price=${analysis['current_price']:.2f}, Volume={analysis['volume_ratio']}x")
                
                logger.info(f"✅ {symbol}: Sending to GPT-4o-mini analysis...")
                
                ai_result = TradeAnalyzer.get_ai_analysis(analysis)
                
                logger.info(f"🎯 {symbol}: GPT Signal = {ai_result['signal']}")
                
                if ai_result['signal'] in ['LONG', 'SHORT']:
                    self.trade_count_today += 1
                    logger.info(f"🎉 {symbol}: VALID {ai_result['signal']} SIGNAL - Trade #{self.trade_count_today}")
                    
                    await self.send_alert(context, symbol, analysis, ai_result)
                else:
                    logger.info(f"⏭️ {symbol}: {ai_result['signal']} - {ai_result.get('reason', '')[:80]}")
                
            except Exception as e:
                logger.error(f"💥 Error scanning {symbol}: {e}", exc_info=True)
            
            await asyncio.sleep(3)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ SCAN COMPLETE - Trades today: {self.trade_count_today}/{MAX_TRADES_PER_DAY}")
        logger.info(f"{'='*80}\n")
    
    async def send_alert(self, context: ContextTypes.DEFAULT_TYPE, symbol: str, analysis: Dict, ai_result: Dict):
        
        logger.info(f"📤 Sending alert for {symbol} with GPT analysis...")
        
        try:
            chart_buf = ChartGenerator.create_chart(analysis['df_30m'], analysis, ai_result, symbol)
        except Exception as e:
            logger.error(f"❌ Chart generation error: {e}")
            chart_buf = None
        
        patterns_text = ", ".join([p['name'] for p in analysis['patterns'][:3]]) if analysis.get('patterns') else "None"
        
        oi_emoji = "📈" if "increasing" in analysis['oi_trend']['trend'] else "📉" if "decreasing" in analysis['oi_trend']['trend'] else "➡️"
        signal_emoji = "🟢" if ai_result['signal'] == 'LONG' else "🔴"
        
        try:
            if ai_result.get('entry') and ai_result.get('sl'):
                rr = abs((ai_result['target'] - ai_result['entry']) / (ai_result['entry'] - ai_result['sl']))
            else:
                rr = 0
        except Exception:
            rr = 0
        
        message = f"""{signal_emoji} **{analysis['symbol']} - {ai_result['signal']} SETUP**

🤖 **GPT-4o-mini Analysis:**
📊 Pattern: {ai_result['pattern']}
💰 Current Price: ${analysis['current_price']:.2f}

🎯 **TRADE DETAILS:**
├ Entry: ${ai_result['entry']:.2f if ai_result.get('entry') else 'N/A'}
├ Stop Loss: ${ai_result['sl']:.2f if ai_result.get('sl') else 'N/A'}
├ Target: ${ai_result['target']:.2f if ai_result.get('target') else 'N/A'}
└ R:R = 1:{rr:.1f}

✅ **CONFIRMATIONS:**
├ Patterns: {patterns_text}
├ Volume: {analysis['volume_ratio']}x avg
└ OI: {oi_emoji} {analysis['oi_trend']['trend']} ({analysis['oi_trend']['change']:+.1f}%)

📈 **SUPPORT & RESISTANCE:**
├ 30m Support: ${analysis['swing_lows_30m'][-1]['price']:.2f if analysis['swing_lows_30m'] else 'N/A'}
├ 30m Resistance: ${analysis['swing_highs_30m'][-1]['price']:.2f if analysis['swing_highs_30m'] else 'N/A'}
├ 4H Support: ${analysis['support_4h']:.2f if analysis['support_4h'] else 'N/A'}
└ 4H Resistance: ${analysis['resistance_4h']:.2f if analysis['resistance_4h'] else 'N/A'}

💡 **GPT Reasoning:** 
{ai_result['reason'][:200]}

⚠️ Trade #{self.trade_count_today}/{MAX_TRADES_PER_DAY} today
"""
        
        try:
            if chart_buf:
                await context.bot.send_photo(
                    chat_id=CHAT_ID,
                    photo=chart_buf,
                    caption=message,
                    parse_mode='Markdown'
                )
            else:
                await context.bot.send_message(
                    chat_id=CHAT_ID,
                    text=message,
                    parse_mode='Markdown'
                )
            
            logger.info(f"✅ Alert sent for {symbol}: {ai_result['signal']}")
            
        except Exception as e:
            logger.error(f"❌ Error sending alert: {e}")
    
    async def send_startup_alert(self, context: ContextTypes.DEFAULT_TYPE):
        startup_message = f"""🤖 **TRADING BOT STARTED**

✅ Status: Online and Active
🕐 Started: {self.bot_start_time.strftime('%Y-%m-%d %H:%M:%S')}

📊 **Configuration:**
├ Symbols: {', '.join(SYMBOLS)}
├ Timeframes: 30m, 1h, 4h
├ Candles per TF: {CANDLE_COUNT}
├ Max Trades/Day: {MAX_TRADES_PER_DAY}
└ Scan Interval: Every 30 minutes

🔧 **Systems:**
├ ✅ Deribit API Connected
├ ✅ OpenAI GPT-4o-mini Ready
├ ✅ In-Memory OI Tracking Active
└ ✅ Telegram Bot Online

🚀 First scan will start in 10 seconds...
"""
        
        try:
            await context.bot.send_message(
                chat_id=CHAT_ID,
                text=startup_message,
                parse_mode='Markdown'
            )
            logger.info("✅ Startup alert sent to Telegram")
        except Exception as e:
            logger.error(f"❌ Failed to send startup alert: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **Trading Bot Active!**\n\n"
        "📊 **Tracking:** BTC & ETH (Deribit)\n"
        "⏱ **Timeframes:** 30m, 1hr, 4hr (500 candles each)\n"
        "🎯 **Max Trades:** 8 per day\n"
        "📈 **Scan Interval:** Every 30 minutes\n"
        "🤖 **AI:** GPT-4o-mini analysis\n\n"
        "**Commands:**\n"
        "/status - Check bot status\n"
        "/scan - Manual scan now\n"
        "/analyze BTC - Analyze specific symbol\n\n"
        "🚀 Bot will automatically scan and alert on valid setups!",
        parse_mode='Markdown'
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot_data.get('trading_bot')
    if bot:
        uptime = datetime.now() - bot.bot_start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        
        await update.message.reply_text(
            f"📊 **Bot Status:**\n\n"
            f"✅ Active and Running\n"
            f"⏰ Uptime: {hours}h {minutes}m\n"
            f"📈 Trades Today: {bot.trade_count_today}/{MAX_TRADES_PER_DAY}\n"
            f"⏱ Scan Interval: 30 minutes\n"
            f"💾 Using In-Memory OI tracking\n"
            f"🤖 AI: GPT-4o-mini\n"
            f"📊 Candles per TF: {CANDLE_COUNT}\n\n"
            f"Next scan in ~30 mins",
            parse_mode='Markdown'
        )

async def scan_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Starting manual scan with GPT-4o-mini analysis...")
    bot = context.bot_data.get('trading_bot')
    if bot:
        await bot.scan_markets(context)
        await update.message.reply_text("✅ Scan complete!")

async def analyze_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /analyze BTC or /analyze ETH")
        return
    
    symbol_input = context.args[0].upper()
    symbol = f"{symbol_input}-PERPETUAL"
    
    if symbol not in SYMBOLS:
        await update.message.reply_text(f"❌ Invalid symbol. Use: BTC or ETH")
        return
    
    await update.message.reply_text(f"🔍 Analyzing {symbol} with GPT-4o-mini...")
    
    try:
        analysis = TradeAnalyzer.analyze_setup(symbol)
        
        if not analysis.get('valid'):
            await update.message.reply_text(f"❌ Cannot analyze {symbol}: {analysis.get('reason', 'Unknown error')}")
            return
        
        ai_result = TradeAnalyzer.get_ai_analysis(analysis)
        
        chart_buf = ChartGenerator.create_chart(analysis['df_30m'], analysis, ai_result, symbol)
        
        patterns_text = ", ".join([p['name'] for p in analysis['patterns']]) if analysis['patterns'] else "None"
        
        message = f"""📊 **{symbol} GPT-4o-mini Analysis**

💰 Price: ${analysis['current_price']:.2f}
📈 Patterns: {patterns_text}
📊 Volume: {analysis['volume_ratio']}x avg
🔄 OI: {analysis['oi_trend']['trend']} ({analysis['oi_trend']['change']:+.1f}%)

🤖 **GPT Signal:** {ai_result['signal']}
💡 **Reasoning:** 
{ai_result.get('reason', 'No reason provided')[:300]}

**Full GPT Response:**
{ai_result.get('full_response', '')[:500]}
"""
        
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=chart_buf,
            caption=message,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"❌ Error in analyze command: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error analyzing {symbol}: {str(e)}")

def main():
    logger.info("="*80)
    logger.info("🚀 INITIALIZING CRYPTO TRADING BOT WITH GPT-4O-MINI")
    logger.info("="*80)
    
    if not TELEGRAM_TOKEN or not OPENAI_API_KEY or not CHAT_ID:
        logger.error("❌ Missing required environment variables!")
        logger.error("Required: TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, TELEGRAM_CHAT_ID")
        return
    
    logger.info("✅ Environment variables validated")
    logger.info("✅ Using in-memory storage (Redis not required)")
    logger.info("✅ GPT-4o-mini configured for trade analysis")
    
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    trading_bot = TradingBot()
    application.bot_data['trading_bot'] = trading_bot
    
    logger.info("✅ Trading bot instance created")
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("scan", scan_now))
    application.add_handler(CommandHandler("analyze", analyze_symbol))
    
    logger.info("✅ Command handlers registered")
    
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(
            trading_bot.scan_markets,
            interval=1800,
            first=10
        )
        logger.info("✅ Job queue configured - scanning every 30 mins")
        
        job_queue.run_once(
            trading_bot.send_startup_alert,
            when=2
        )
        logger.info("✅ Startup alert scheduled")
    else:
        logger.warning("⚠️ Job queue not available - install with: pip install 'python-telegram-bot[job-queue]'")
        logger.info("ℹ️ You can still use manual commands: /scan and /analyze")
    
    logger.info("="*80)
    logger.info("🚀 BOT STARTING...")
    logger.info(f"📊 Tracking: {', '.join(SYMBOLS)}")
    logger.info(f"⏱ Scan interval: 30 minutes (if job-queue installed)")
    logger.info(f"📈 Candles per TF: {CANDLE_COUNT}")
    logger.info(f"🎯 Max trades per day: {MAX_TRADES_PER_DAY}")
    logger.info(f"🤖 AI Engine: GPT-4o-mini")
    logger.info("="*80)
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
