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
import io

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
MAX_TRADES_PER_DAY = 8
CANDLE_COUNT = 500

class DeribitClient:
    RESOLUTION_MAP = {'30': '30', '60': '60', '240': '1D'}
    
    @staticmethod
    def get_candles(symbol: str, timeframe: str, count: int = CANDLE_COUNT) -> pd.DataFrame:
        logger.info(f"Fetching {count} candles for {symbol} {timeframe}m...")
        resolution = DeribitClient.RESOLUTION_MAP.get(timeframe, timeframe)
        url = f"{DERIBIT_BASE}/get_tradingview_chart_data"
        tf_minutes = int(timeframe)
        days_needed = count + 10 if timeframe == '240' else (count * tf_minutes) // (24 * 60) + 10
        
        params = {
            'instrument_name': symbol,
            'resolution': resolution,
            'start_timestamp': int((datetime.now() - timedelta(days=days_needed)).timestamp() * 1000),
            'end_timestamp': int(datetime.now().timestamp() * 1000)
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                return pd.DataFrame()
            
            data = response.json()
            if 'result' not in data or data['result'].get('status') != 'ok':
                return pd.DataFrame()
            
            result = data['result']
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(result['ticks'], unit='ms'),
                'open': result['open'],
                'high': result['high'],
                'low': result['low'],
                'close': result['close'],
                'volume': result['volume']
            })
            df.set_index('timestamp', inplace=True)
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df.tail(count)
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_order_book(symbol: str) -> Dict:
        url = f"{DERIBIT_BASE}/get_order_book"
        params = {'instrument_name': symbol, 'depth': 10}
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if 'result' in data:
                return {
                    'open_interest': data['result'].get('open_interest', 0),
                    'volume_24h': data['result'].get('stats', {}).get('volume', 0),
                    'mark_price': data['result'].get('mark_price', 0)
                }
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
        return {'open_interest': 0, 'volume_24h': 0, 'mark_price': 0}

class TechnicalAnalyzer:
    @staticmethod
    def find_swing_points(df: pd.DataFrame, period: int = 5) -> Tuple[List, List]:
        swing_highs = []
        swing_lows = []
        for i in range(period, len(df) - period):
            if df['high'].iloc[i] == df['high'].iloc[i-period:i+period+1].max():
                swing_highs.append({'price': df['high'].iloc[i], 'index': i, 'timestamp': df.index[i]})
            if df['low'].iloc[i] == df['low'].iloc[i-period:i+period+1].min():
                swing_lows.append({'price': df['low'].iloc[i], 'index': i, 'timestamp': df.index[i]})
        
        current_price = df['close'].iloc[-1]
        relevant_highs = [s for s in swing_highs if s['index'] >= len(df) - 50 and s['price'] >= current_price and s['price'] <= current_price * 1.05]
        relevant_lows = [s for s in swing_lows if s['index'] >= len(df) - 50 and s['price'] <= current_price and s['price'] >= current_price * 0.95]
        
        if not relevant_highs:
            relevant_highs = sorted(swing_highs, key=lambda x: abs(x['price'] - current_price))[:3]
        if not relevant_lows:
            relevant_lows = sorted(swing_lows, key=lambda x: abs(x['price'] - current_price))[:3]
        
        return sorted(relevant_highs, key=lambda x: x['index'], reverse=True)[:3], sorted(relevant_lows, key=lambda x: x['index'], reverse=True)[:3]
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> List[Dict]:
        patterns = []
        if len(df) < 3:
            return patterns
        last = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(last['close'] - last['open'])
        candle_range = last['high'] - last['low']
        if candle_range > 0 and body < candle_range * 0.1:
            patterns.append({'name': 'Doji', 'signal': 'neutral'})
        return patterns

class OITracker:
    @staticmethod
    def store_oi(symbol: str, oi_data: Dict):
        timestamp = int(datetime.now().timestamp())
        if symbol not in oi_storage:
            oi_storage[symbol] = []
        oi_storage[symbol].append({'timestamp': timestamp, 'data': oi_data})
        cutoff = timestamp - (3 * 3600)
        oi_storage[symbol] = [e for e in oi_storage[symbol] if e['timestamp'] >= cutoff]
    
    @staticmethod
    def analyze_oi_trend(symbol: str) -> Dict:
        if symbol not in oi_storage or len(oi_storage[symbol]) < 2:
            return {'trend': 'insufficient_data', 'change': 0}
        history = oi_storage[symbol]
        current_oi = history[-1]['data']['open_interest']
        old_oi = history[0]['data']['open_interest']
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
        return {'trend': trend, 'change': round(change, 2), 'current_oi': current_oi, 'previous_oi': old_oi}

class ChartGenerator:
    @staticmethod
    def create_chart(df: pd.DataFrame, analysis: Dict, ai_result: Dict, symbol: str) -> io.BytesIO:
        mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit', volume='in', alpha=0.9)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', gridcolor='#e0e0e0', facecolor='white', figcolor='white')
        
        fig, axes = mpf.plot(df.tail(100), type='candle', style=s, volume=True, returnfig=True, figsize=(14, 8), title=f"{symbol} - {ai_result.get('pattern', 'Analysis')}")
        ax = axes[0]
        
        if analysis.get('swing_lows_30m'):
            for swing in analysis['swing_lows_30m'][-3:]:
                ax.axhline(y=swing['price'], color='#4caf50', linestyle='--', linewidth=1.5, alpha=0.7)
        
        if analysis.get('swing_highs_30m'):
            for swing in analysis['swing_highs_30m'][-3:]:
                ax.axhline(y=swing['price'], color='#f44336', linestyle='--', linewidth=1.5, alpha=0.7)
        
        current_price = df['close'].iloc[-1]
        ax.axhline(y=current_price, color='#ff9800', linestyle=':', linewidth=2)
        
        if ai_result.get('entry') and ai_result['signal'] in ['LONG', 'SHORT']:
            ax.axhline(y=ai_result['entry'], color='#2196f3', linestyle='-.', linewidth=2.5)
            if ai_result.get('sl'):
                ax.axhline(y=ai_result['sl'], color='#d32f2f', linestyle='-.', linewidth=2)
            if ai_result.get('target'):
                ax.axhline(y=ai_result['target'], color='#388e3c', linestyle='-.', linewidth=2)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='white')
        buf.seek(0)
        plt.close(fig)
        return buf

class TradeAnalyzer:
    @staticmethod
    def analyze_setup(symbol: str) -> Dict:
        df_30m = DeribitClient.get_candles(symbol, '30', CANDLE_COUNT)
        df_1h = DeribitClient.get_candles(symbol, '60', CANDLE_COUNT)
        df_4h = DeribitClient.get_candles(symbol, '240', CANDLE_COUNT)
        
        if df_30m.empty or df_1h.empty or df_4h.empty:
            return {'valid': False, 'symbol': symbol}
        
        oi_data = DeribitClient.get_order_book(symbol)
        OITracker.store_oi(symbol, oi_data)
        oi_trend = OITracker.analyze_oi_trend(symbol)
        
        swing_highs_30m, swing_lows_30m = TechnicalAnalyzer.find_swing_points(df_30m)
        swing_highs_4h, swing_lows_4h = TechnicalAnalyzer.find_swing_points(df_4h)
        patterns = TechnicalAnalyzer.detect_patterns(df_30m)
        
        current_price = df_30m['close'].iloc[-1]
        volume_ratio = df_30m['volume'].iloc[-1] / df_30m['volume'].tail(20).mean() if df_30m['volume'].tail(20).mean() > 0 else 0
        
        resistance_4h = min([s['price'] for s in swing_highs_4h if s['price'] >= current_price], default=None)
        support_4h = max([s['price'] for s in swing_lows_4h if s['price'] <= current_price], default=None)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'patterns': patterns,
            'volume_ratio': round(volume_ratio, 2),
            'oi_trend': oi_trend,
            'swing_highs_30m': swing_highs_30m,
            'swing_lows_30m': swing_lows_30m,
            'resistance_4h': resistance_4h,
            'support_4h': support_4h,
            'df_30m': df_30m,
            'valid': True
        }
    
    @staticmethod
    def get_ai_analysis(analysis: Dict) -> Dict:
        patterns_text = ', '.join([p['name'] for p in analysis.get('patterns', [])]) or 'None'
        prompt = f"""Analyze {analysis['symbol']}:
Price: ${analysis['current_price']:.2f}
Patterns: {patterns_text}
Volume: {analysis['volume_ratio']}x
OI Trend: {analysis['oi_trend']['trend']}

Decide: LONG, SHORT, or NO_TRADE
If tradeable, provide ENTRY, SL, TARGET, REASON"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a crypto trader. Be conservative."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            ai_text = response.choices[0].message.content
            result = {'signal': 'NO_TRADE', 'entry': None, 'sl': None, 'target': None, 'pattern': 'None', 'reason': ai_text}
            
            for line in ai_text.split('\n'):
                line = line.strip()
                if 'SIGNAL:' in line:
                    if 'LONG' in line.upper():
                        result['signal'] = 'LONG'
                    elif 'SHORT' in line.upper():
                        result['signal'] = 'SHORT'
                elif 'ENTRY:' in line:
                    try:
                        val = line.split('ENTRY:')[-1].strip().replace('$', '').replace(',', '')
                        result['entry'] = float(val.split()[0])
                    except:
                        pass
                elif 'SL:' in line or 'STOP' in line:
                    try:
                        val = line.split(':')[-1].strip().replace('$', '').replace(',', '')
                        result['sl'] = float(val.split()[0])
                    except:
                        pass
                elif 'TARGET:' in line:
                    try:
                        val = line.split('TARGET:')[-1].strip().replace('$', '').replace(',', '')
                        result['target'] = float(val.split()[0])
                    except:
                        pass
            
            return result
        except Exception as e:
            logger.error(f"GPT error: {e}")
            return {'signal': 'ERROR', 'reason': str(e), 'entry': None, 'sl': None, 'target': None, 'pattern': 'Error'}

class TradingBot:
    def __init__(self):
        self.trade_count_today = 0
        self.last_reset = datetime.now().date()
        self.bot_start_time = datetime.now()
    
    def reset_daily_counter(self):
        if datetime.now().date() > self.last_reset:
            self.trade_count_today = 0
            self.last_reset = datetime.now().date()
    
    async def scan_markets(self, context: ContextTypes.DEFAULT_TYPE):
        logger.info("STARTING SCAN")
        self.reset_daily_counter()
        
        if self.trade_count_today >= MAX_TRADES_PER_DAY:
            return
        
        for symbol in SYMBOLS:
            try:
                analysis = TradeAnalyzer.analyze_setup(symbol)
                if not analysis.get('valid'):
                    continue
                
                ai_result = TradeAnalyzer.get_ai_analysis(analysis)
                
                if ai_result['signal'] in ['LONG', 'SHORT']:
                    self.trade_count_today += 1
                    await self.send_alert(context, symbol, analysis, ai_result)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
            await asyncio.sleep(3)
    
    async def send_alert(self, context: ContextTypes.DEFAULT_TYPE, symbol: str, analysis: Dict, ai_result: Dict):
        try:
            chart_buf = ChartGenerator.create_chart(analysis['df_30m'], analysis, ai_result, symbol)
        except Exception as e:
            chart_buf = None
        
        patterns_text = ', '.join([p['name'] for p in analysis['patterns']]) if analysis.get('patterns') else 'None'
        
        message = f"""**{symbol} - {ai_result['signal']}**

Price: ${analysis['current_price']:.2f}
Entry: ${ai_result.get('entry', 0):.2f}
SL: ${ai_result.get('sl', 0):.2f}
Target: ${ai_result.get('target', 0):.2f}

Patterns: {patterns_text}
Volume: {analysis['volume_ratio']}x
OI: {analysis['oi_trend']['trend']}

Reason: {ai_result.get('reason', '')[:150]}
"""
        
        try:
            if chart_buf:
                await context.bot.send_photo(chat_id=CHAT_ID, photo=chart_buf, caption=message, parse_mode='Markdown')
            else:
                await context.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Alert send error: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot Active! Use /scan or /analyze BTC")

async def scan_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Scanning...")
    bot = context.bot_data.get('trading_bot')
    if bot:
        await bot.scan_markets(context)
        await update.message.reply_text("Scan complete!")

async def analyze_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /analyze BTC or ETH")
        return
    
    symbol = f"{context.args[0].upper()}-PERPETUAL"
    if symbol not in SYMBOLS:
        await update.message.reply_text("Invalid symbol")
        return
    
    try:
        analysis = TradeAnalyzer.analyze_setup(symbol)
        if not analysis.get('valid'):
            await update.message.reply_text("Cannot analyze")
            return
        
        ai_result = TradeAnalyzer.get_ai_analysis(analysis)
        chart_buf = ChartGenerator.create_chart(analysis['df_30m'], analysis, ai_result, symbol)
        
        message = f"""{symbol}: {ai_result['signal']}
Price: ${analysis['current_price']:.2f}
{ai_result.get('reason', '')[:200]}"""
        
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=chart_buf, caption=message)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

def main():
    logger.info("STARTING BOT")
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    trading_bot = TradingBot()
    application.bot_data['trading_bot'] = trading_bot
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("scan", scan_now))
    application.add_handler(CommandHandler("analyze", analyze_symbol))
    
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(trading_bot.scan_markets, interval=1800, first=10)
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
