import os
import time
import asyncio
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import mplfinance as mpf
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from PIL import Image
import json
import base64
from io import BytesIO

# ==================== CONFIG ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Binance altcoins (Perpetual futures)
BINANCE_COINS = ["LINK", "DOGE", "XRP", "BNB", "LTC", "TRX", "ADA", "AVAX", "SOL"]

# Deribit options (BTC, ETH only)
DERIBIT_OPTIONS = ["BTC", "ETH"]

ALL_COINS = BINANCE_COINS + DERIBIT_OPTIONS
SCAN_INTERVAL = 3600  # 1 hour (3600 seconds)
ANALYSIS_ACCURACY_THRESHOLD = 70  # Minimum 70% confidence

print("ℹ️ AI-Powered Trading Scanner: GPT-4o Mini + Vision Analysis")

# ==================== BINANCE DATA FETCH ====================
async def fetch_binance_candles(session, symbol, interval="1h", limit=1000):
    """Fetch 4 months data from Binance using multiple requests"""
    all_data = []
    end_time = int(time.time() * 1000)
    
    for i in range(3):
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": f"{symbol}USDT",
            "interval": interval,
            "limit": 1000,
            "endTime": end_time
        }
        
        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if not data:
                    break
                
                all_data = data + all_data
                end_time = int(data[0][0]) - 1
                
                await asyncio.sleep(0.2)
        except Exception as e:
            print(f"❌ Binance error for {symbol}: {e}")
            break
    
    if all_data:
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        four_months_ago = datetime.now() - timedelta(days=120)
        df = df[df.index >= four_months_ago]
        
        print(f"✅ Binance: Fetched {len(df)} candles for {symbol}")
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    return None

async def fetch_binance_ticker(session, symbol):
    """Fetch 24h ticker data from Binance"""
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    params = {"symbol": f"{symbol}USDT"}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            return {
                'last_price': float(data['lastPrice']),
                'price_change_24h': float(data['priceChangePercent']),
                '24h_high': float(data['highPrice']),
                '24h_low': float(data['lowPrice']),
                'volume_24h': float(data['volume']),
                'volume_usd': float(data['quoteVolume'])
            }
    except Exception as e:
        print(f"❌ Binance ticker error: {e}")
        return None

async def fetch_binance_funding_rate(session, symbol):
    """Fetch funding rate from Binance"""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": f"{symbol}USDT", "limit": 1}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            if data:
                return float(data[0]['fundingRate']) * 100
    except Exception as e:
        print(f"❌ Funding rate error: {e}")
    return 0

async def fetch_binance_open_interest(session, symbol):
    """Fetch open interest from Binance"""
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    params = {"symbol": f"{symbol}USDT"}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            return float(data['openInterest'])
    except Exception as e:
        print(f"❌ OI error: {e}")
    return 0

# ==================== DERIBIT DATA FETCH ====================
async def fetch_deribit_candles(session, symbol):
    """Fetch 4 months data from Deribit"""
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    
    end_timestamp = int(time.time() * 1000)
    start_timestamp = int((datetime.now() - timedelta(days=120)).timestamp() * 1000)
    
    params = {
        "instrument_name": f"{symbol}-PERPETUAL",
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "resolution": "60"
    }
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            if data.get("result") and data["result"].get("status") == "ok":
                result = data["result"]
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(result['ticks'], unit='ms'),
                    'open': result['open'],
                    'high': result['high'],
                    'low': result['low'],
                    'close': result['close'],
                    'volume': result['volume']
                })
                df.set_index('timestamp', inplace=True)
                print(f"✅ Deribit: Fetched {len(df)} candles for {symbol}")
                return df
    except Exception as e:
        print(f"❌ Deribit candles error: {e}")
    return None

async def fetch_deribit_ticker(session, symbol):
    """Fetch ticker from Deribit"""
    url = "https://www.deribit.com/api/v2/public/ticker"
    params = {"instrument_name": f"{symbol}-PERPETUAL"}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            if data.get("result"):
                result = data["result"]
                return {
                    'last_price': result.get('last_price', 0),
                    'mark_price': result.get('mark_price', 0),
                    'index_price': result.get('index_price', 0),
                    'volume_24h': result.get('stats', {}).get('volume', 0),
                    'volume_usd': result.get('stats', {}).get('volume_usd', 0),
                    'open_interest': result.get('open_interest', 0),
                    'funding_8h': result.get('funding_8h', 0),
                    '24h_high': result.get('stats', {}).get('high', 0),
                    '24h_low': result.get('stats', {}).get('low', 0),
                    'price_change_24h': result.get('stats', {}).get('price_change', 0)
                }
    except Exception as e:
        print(f"❌ Deribit ticker error: {e}")
    return None

# ==================== CHART GENERATION ====================
def create_chart(df, symbol, source="Binance"):
    """Create ultra-wide HD candlestick chart"""
    chart_file = f"chart_{symbol}_{int(time.time())}.png"
    
    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350', 
        edge='inherit', 
        wick={'up':'#26a69a', 'down':'#ef5350'}, 
        volume='in', alpha=0.9
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc, gridstyle='-', 
        gridcolor='#e0e0e0', gridaxis='both',
        facecolor='white', figcolor='white',
        edgecolor='#cccccc', 
        rc={'font.size': 12, 'axes.linewidth': 1.5},
        y_on_right=True
    )
    
    title = f"{symbol}USDT ({source}) | Last 4 Months (1H) | {len(df)} Candles"
    
    try:
        mpf.plot(
            df, type='candle', style=s, title=title,
            ylabel='Price (USDT)', volume=True, 
            savefig=dict(fname=chart_file, dpi=150, bbox_inches='tight'),
            figsize=(32, 14),
            warn_too_much_data=len(df)+1
        )
        print(f"✅ Chart created: {chart_file}")
        return chart_file
    except Exception as e:
        print(f"❌ Chart error: {e}")
    return None

# ==================== TECHNICAL INDICATORS ====================
def calculate_technical_indicators(df):
    """Calculate technical indicators for analysis"""
    try:
        # Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Support & Resistance
        recent_data = df.tail(100)
        df['support'] = recent_data['low'].min()
        df['resistance'] = recent_data['high'].max()
        
        return df
    except Exception as e:
        print(f"❌ Indicator calculation error: {e}")
        return df

def prepare_analysis_data(df, ticker, funding_rate, oi, symbol):
    """Prepare comprehensive data for AI analysis"""
    df = calculate_technical_indicators(df)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    analysis_data = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "current_price": float(latest['close']),
        "price_change_24h": ticker['price_change_24h'] if ticker else 0,
        "volume_24h": ticker['volume_usd'] if ticker else 0,
        "funding_rate": funding_rate,
        "open_interest": oi,
        
        "technical_indicators": {
            "sma_20": float(latest['SMA_20']) if pd.notna(latest['SMA_20']) else None,
            "sma_50": float(latest['SMA_50']) if pd.notna(latest['SMA_50']) else None,
            "sma_200": float(latest['SMA_200']) if pd.notna(latest['SMA_200']) else None,
            "rsi": float(latest['RSI']) if pd.notna(latest['RSI']) else None,
            "macd": float(latest['MACD']) if pd.notna(latest['MACD']) else None,
            "macd_signal": float(latest['MACD_signal']) if pd.notna(latest['MACD_signal']) else None,
            "bb_upper": float(latest['BB_upper']) if pd.notna(latest['BB_upper']) else None,
            "bb_lower": float(latest['BB_lower']) if pd.notna(latest['BB_lower']) else None,
            "support": float(latest['support']) if pd.notna(latest['support']) else None,
            "resistance": float(latest['resistance']) if pd.notna(latest['resistance']) else None
        },
        
        "candlestick_data": {
            "last_10_candles": df[['open', 'high', 'low', 'close', 'volume']].tail(10).to_dict('records')
        }
    }
    
    return analysis_data

# ==================== GPT-4O MINI VISION ANALYSIS ====================
async def analyze_with_gpt4o_mini(chart_file, analysis_data, session):
    """Analyze chart using GPT-4o Mini Vision"""
    try:
        # Convert image to base64
        with open(chart_file, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare prompt
        prompt = f"""You are an expert crypto trader analyzing {analysis_data['symbol']}USDT.

**CURRENT DATA:**
- Price: ${analysis_data['current_price']:,.2f}
- 24h Change: {analysis_data['price_change_24h']:.2f}%
- Volume (24h): ${analysis_data['volume_24h']:,.0f}
- Funding Rate: {analysis_data['funding_rate']:.4f}%
- Open Interest: {analysis_data['open_interest']:,.2f}

**TECHNICAL INDICATORS:**
{json.dumps(analysis_data['technical_indicators'], indent=2)}

**YOUR TASK:**
Analyze the attached 4-month chart and provide:

1. **Candlestick Patterns:** Identify key patterns (Doji, Hammer, Engulfing, etc.)
2. **Chart Patterns:** Head & Shoulders, Double Top/Bottom, Triangles, Flags, Wedges
3. **Price Action:** Trend direction, momentum, breakouts
4. **Support & Resistance:** Key levels with exact prices
5. **Trend Lines:** Draw trend lines and identify channels

**TRADING SIGNALS:**
- **SHORT TERM (1-7 days):** Entry, Stop Loss, Take Profit, Position Size
- **MEDIUM TERM (1-4 weeks):** Entry, Stop Loss, Take Profit, Position Size

**CONFIDENCE SCORE:** Rate your analysis accuracy from 0-100%

**OUTPUT FORMAT (JSON):**
{{
  "confidence_score": 85,
  "trend": "BULLISH/BEARISH/NEUTRAL",
  "candlestick_patterns": ["pattern1", "pattern2"],
  "chart_patterns": ["pattern1", "pattern2"],
  "support_levels": [12.50, 11.80],
  "resistance_levels": [14.20, 15.00],
  "short_term_trade": {{
    "signal": "LONG/SHORT/NONE",
    "entry": 13.50,
    "stop_loss": 12.80,
    "take_profit": 15.20,
    "risk_reward": 2.5,
    "position_size": "2-3%"
  }},
  "medium_term_trade": {{
    "signal": "LONG/SHORT/NONE",
    "entry": 13.00,
    "stop_loss": 11.50,
    "take_profit": 17.00,
    "risk_reward": 3.0,
    "position_size": "5-7%"
  }},
  "reasoning": "Detailed explanation of your analysis"
}}

IMPORTANT: Only provide trades if confidence >= 70%. Be conservative."""

        # Call OpenAI API
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            ai_response = result['choices'][0]['message']['content']
            
            # Extract JSON from response
            if "```json" in ai_response:
                json_str = ai_response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = ai_response
            
            analysis_result = json.loads(json_str)
            print(f"✅ AI Analysis completed for {analysis_data['symbol']}")
            return analysis_result
            
    except Exception as e:
        print(f"❌ GPT-4o Mini analysis error: {e}")
        return None

# ==================== TELEGRAM ALERT ====================
async def send_trade_alert(bot, symbol, analysis_result, chart_file):
    """Send trading alert if confidence >= 70%"""
    try:
        confidence = analysis_result.get('confidence_score', 0)
        
        if confidence < ANALYSIS_ACCURACY_THRESHOLD:
            print(f"⚠️ {symbol}: Confidence {confidence}% < 70%, skipping alert")
            return
        
        # Send chart
        with open(chart_file, 'rb') as photo:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo,
                caption=f"🤖 **AI Trading Signal: {symbol}**\n📊 Confidence: {confidence}%"
            )
        
        # Format alert message
        msg = f"🎯 **{symbol} TRADING SIGNAL**\n\n"
        msg += f"🔮 **Confidence:** {confidence}%\n"
        msg += f"📈 **Trend:** {analysis_result.get('trend', 'N/A')}\n\n"
        
        # Patterns
        if analysis_result.get('candlestick_patterns'):
            msg += f"🕯️ **Candlestick Patterns:**\n"
            for pattern in analysis_result['candlestick_patterns']:
                msg += f"   • {pattern}\n"
            msg += "\n"
        
        if analysis_result.get('chart_patterns'):
            msg += f"📊 **Chart Patterns:**\n"
            for pattern in analysis_result['chart_patterns']:
                msg += f"   • {pattern}\n"
            msg += "\n"
        
        # Support & Resistance
        if analysis_result.get('support_levels'):
            msg += f"🛡️ **Support Levels:**\n"
            for level in analysis_result['support_levels']:
                msg += f"   • ${level:,.2f}\n"
            msg += "\n"
        
        if analysis_result.get('resistance_levels'):
            msg += f"⚔️ **Resistance Levels:**\n"
            for level in analysis_result['resistance_levels']:
                msg += f"   • ${level:,.2f}\n"
            msg += "\n"
        
        # Short Term Trade
        st = analysis_result.get('short_term_trade', {})
        if st.get('signal') and st['signal'] != 'NONE':
            msg += f"⚡ **SHORT TERM (1-7 Days):**\n"
            msg += f"   Signal: **{st['signal']}**\n"
            msg += f"   Entry: ${st.get('entry', 0):,.2f}\n"
            msg += f"   Stop Loss: ${st.get('stop_loss', 0):,.2f}\n"
            msg += f"   Take Profit: ${st.get('take_profit', 0):,.2f}\n"
            msg += f"   Risk/Reward: {st.get('risk_reward', 0):.1f}\n"
            msg += f"   Position: {st.get('position_size', 'N/A')}\n\n"
        
        # Medium Term Trade
        mt = analysis_result.get('medium_term_trade', {})
        if mt.get('signal') and mt['signal'] != 'NONE':
            msg += f"📅 **MEDIUM TERM (1-4 Weeks):**\n"
            msg += f"   Signal: **{mt['signal']}**\n"
            msg += f"   Entry: ${mt.get('entry', 0):,.2f}\n"
            msg += f"   Stop Loss: ${mt.get('stop_loss', 0):,.2f}\n"
            msg += f"   Take Profit: ${mt.get('take_profit', 0):,.2f}\n"
            msg += f"   Risk/Reward: {mt.get('risk_reward', 0):.1f}\n"
            msg += f"   Position: {mt.get('position_size', 'N/A')}\n\n"
        
        # Reasoning
        msg += f"💡 **Analysis:**\n{analysis_result.get('reasoning', 'N/A')}\n\n"
        msg += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send message
        if len(msg) > 4000:
            chunks = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
            for chunk in chunks:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=chunk,
                    parse_mode='Markdown'
                )
        else:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
        
        print(f"✅ Trade alert sent for {symbol}")
        
    except Exception as e:
        print(f"❌ Alert error: {e}")

# ==================== MAIN SCANNER ====================
async def scan_cryptos(bot: Bot):
    """Main AI scanner - runs every 1 hour"""
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"🚀 **AI Trading Scanner Started!**\n\n🤖 Model: GPT-4o Mini\n📊 Coins: {len(ALL_COINS)}\n⏰ Analysis: Every 1 hour\n🎯 Min Confidence: {ANALYSIS_ACCURACY_THRESHOLD}%",
            parse_mode='Markdown'
        )
    except Exception as e:
        print(f"❌ Startup error: {e}")
    
    print(f"🤖 AI Scanner: {len(ALL_COINS)} coins | Every 1 hour | Min confidence: {ANALYSIS_ACCURACY_THRESHOLD}%\n")
    
    async with aiohttp.ClientSession() as session:
        while True:
            # Scan Binance coins
            for symbol in BINANCE_COINS:
                print(f"\n{'='*60}")
                print(f"🔍 Analyzing {symbol} at {datetime.now().strftime('%H:%M:%S')}")
                
                df = await fetch_binance_candles(session, symbol)
                ticker = await fetch_binance_ticker(session, symbol)
                funding = await fetch_binance_funding_rate(session, symbol)
                oi = await fetch_binance_open_interest(session, symbol)
                
                if df is None or df.empty:
                    print(f"⚠️ Skipping {symbol}")
                    continue
                
                # Create chart
                chart_file = create_chart(df, symbol, "Binance")
                if not chart_file:
                    continue
                
                # Prepare analysis data
                analysis_data = prepare_analysis_data(df, ticker, funding, oi, symbol)
                
                # AI Analysis
                analysis_result = await analyze_with_gpt4o_mini(chart_file, analysis_data, session)
                
                if analysis_result:
                    await send_trade_alert(bot, symbol, analysis_result, chart_file)
                
                try:
                    os.remove(chart_file)
                except:
                    pass
                
                await asyncio.sleep(10)
            
            # Scan Deribit BTC/ETH
            for symbol in DERIBIT_OPTIONS:
                print(f"\n{'='*60}")
                print(f"🎯 Analyzing {symbol} (Deribit) at {datetime.now().strftime('%H:%M:%S')}")
                
                df = await fetch_deribit_candles(session, symbol)
                ticker = await fetch_deribit_ticker(session, symbol)
                
                if df is None or df.empty:
                    continue
                
                chart_file = create_chart(df, symbol, "Deribit")
                if not chart_file:
                    continue
                
                analysis_data = prepare_analysis_data(df, ticker, 0, 0, symbol)
                analysis_result = await analyze_with_gpt4o_mini(chart_file, analysis_data, session)
                
                if analysis_result:
                    await send_trade_alert(bot, symbol, analysis_result, chart_file)
                
                try:
                    os.remove(chart_file)
                except:
                    pass
                
                await asyncio.sleep(10)
            
            print(f"\n⏳ Next scan in 1 hour...\n")
            await asyncio.sleep(SCAN_INTERVAL)

# ==================== TELEGRAM COMMANDS ====================
async def start(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **AI Crypto Trading Scanner**\n\n"
        f"🧠 Model: GPT-4o Mini Vision\n"
        f"📊 Coins: {len(ALL_COINS)}\n"
        f"⏰ Analysis: Every 1 hour\n"
        f"🎯 Min Confidence: {ANALYSIS_ACCURACY_THRESHOLD}%\n\n"
        f"Commands: /start /status",
        parse_mode='Markdown'
    )

async def status(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"✅ Running | 🤖 AI Active | 📊 {len(ALL_COINS)} coins | ⏰ Every 1hr",
        parse_mode='Markdown'
    )

# ==================== POST INIT ====================
async def post_init(application: Application):
    print("🚀 Starting AI Trading Scanner...")
    asyncio.create_task(scan_cryptos(application.bot))

# ==================== RUN BOT ====================
def main():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.post_init = post_init
    
    print("🤖 Starting AI Trading Bot...")
    application.run_polling()

if __name__ == "__main__":
    main()
