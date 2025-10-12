import os
import sys
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== CONFIG WITH VALIDATION ====================
def validate_env():
    """Validate all required environment variables"""
    required_vars = {
        "TELEGRAM_BOT_TOKEN": "Get from @BotFather on Telegram",
        "TELEGRAM_CHAT_ID": "Get from @userinfobot on Telegram",
        "OPENAI_API_KEY": "Get from https://platform.openai.com/api-keys"
    }
    
    missing = []
    invalid = []
    
    for var, instruction in required_vars.items():
        value = os.getenv(var)
        
        if not value:
            missing.append(f"❌ {var} is missing")
        elif var == "TELEGRAM_BOT_TOKEN":
            if "ABCdef" in value or value.startswith("123456789:"):
                invalid.append(f"❌ {var} is still the example value!\n   {instruction}")
        elif var == "OPENAI_API_KEY":
            if "abcdefghij" in value or not value.startswith("sk-"):
                invalid.append(f"❌ {var} is invalid!\n   {instruction}")
        elif var == "TELEGRAM_CHAT_ID":
            if not value.isdigit() and not value.startswith("-"):
                invalid.append(f"❌ {var} should be a number!\n   {instruction}")
    
    if missing or invalid:
        print("\n" + "="*60)
        print("🚨 CONFIGURATION ERROR 🚨")
        print("="*60 + "\n")
        
        if missing:
            print("Missing environment variables:")
            for msg in missing:
                print(f"  {msg}")
            print()
        
        if invalid:
            print("Invalid environment variables:")
            for msg in invalid:
                print(f"  {msg}")
            print()
        
        print("📝 Steps to fix:")
        print("  1. Copy .env.example to .env")
        print("  2. Edit .env file with your real API keys")
        print("  3. Save and run again\n")
        
        sys.exit(1)

# Validate before proceeding
validate_env()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Trading pairs configuration
BINANCE_COINS = ["LINK", "DOGE", "XRP", "BNB", "LTC", "TRX", "ADA", "AVAX", "SOL"]
DERIBIT_OPTIONS = ["BTC", "ETH"]  # These have options chain analysis

ALL_COINS = BINANCE_COINS + DERIBIT_OPTIONS
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "3600"))  # 1 hour
ANALYSIS_ACCURACY_THRESHOLD = int(os.getenv("MIN_CONFIDENCE", "70"))

print("✅ Configuration validated successfully!")
print(f"ℹ️ AI-Powered Trading Scanner: GPT-4o Mini + Vision Analysis")
print(f"📊 Binance Coins (Chart Only): {', '.join(BINANCE_COINS)}")
print(f"🎯 Deribit (Chart + Options): {', '.join(DERIBIT_OPTIONS)}")
print(f"⏰ Scan Interval: {SCAN_INTERVAL//60} minutes")
print(f"🎯 Min Confidence: {ANALYSIS_ACCURACY_THRESHOLD}%\n")

# ==================== BINANCE DATA FETCH ====================
async def fetch_binance_candles(session, symbol, interval="1h", limit=1000):
    """Fetch 4 months data from Binance"""
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

async def fetch_deribit_options_chain(session, symbol):
    """Fetch options chain from Deribit (BTC/ETH only)"""
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    params = {"currency": symbol, "kind": "option"}
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            if data.get("result"):
                options = data["result"]
                chain_data = []
                for opt in options:
                    chain_data.append({
                        'instrument': opt['instrument_name'],
                        'type': 'CALL' if '-C' in opt['instrument_name'] else 'PUT',
                        'strike': opt['instrument_name'].split('-')[2] if len(opt['instrument_name'].split('-')) > 2 else 'N/A',
                        'expiry': opt['instrument_name'].split('-')[1] if len(opt['instrument_name'].split('-')) > 1 else 'N/A',
                        'bid_price': opt.get('bid_price', 0),
                        'ask_price': opt.get('ask_price', 0),
                        'volume': opt.get('volume', 0),
                        'open_interest': opt.get('open_interest', 0),
                        'mark_iv': opt.get('mark_iv', 0)
                    })
                
                df = pd.DataFrame(chain_data)
                print(f"✅ Fetched {len(df)} options for {symbol}")
                return df
    except Exception as e:
        print(f"❌ Options error: {e}")
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
    """Calculate technical indicators"""
    try:
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        recent_data = df.tail(100)
        df['support'] = recent_data['low'].min()
        df['resistance'] = recent_data['high'].max()
        
        return df
    except Exception as e:
        print(f"❌ Indicator error: {e}")
        return df

def prepare_analysis_data(df, ticker, funding_rate, oi, symbol):
    """Prepare data for AI analysis"""
    df = calculate_technical_indicators(df)
    
    latest = df.iloc[-1]
    
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

def prepare_options_summary(options_df, current_price):
    """Prepare options chain summary for analysis"""
    if options_df is None or options_df.empty:
        return None
    
    options_df['strike_num'] = pd.to_numeric(options_df['strike'], errors='coerce')
    
    # Filter ATM options (±20%)
    atm_options = options_df[
        (options_df['strike_num'] >= current_price * 0.8) & 
        (options_df['strike_num'] <= current_price * 1.2)
    ].copy()
    
    # Calculate Put/Call Ratio
    call_vol = atm_options[atm_options['type'] == 'CALL']['volume'].sum()
    put_vol = atm_options[atm_options['type'] == 'PUT']['volume'].sum()
    call_oi = atm_options[atm_options['type'] == 'CALL']['open_interest'].sum()
    put_oi = atm_options[atm_options['type'] == 'PUT']['open_interest'].sum()
    
    pcr_volume = (put_vol / call_vol) if call_vol > 0 else 0
    pcr_oi = (put_oi / call_oi) if call_oi > 0 else 0
    
    # Get top strikes by volume
    top_calls = atm_options[atm_options['type'] == 'CALL'].nlargest(5, 'volume')[
        ['strike', 'volume', 'open_interest', 'mark_iv']
    ].to_dict('records')
    
    top_puts = atm_options[atm_options['type'] == 'PUT'].nlargest(5, 'volume')[
        ['strike', 'volume', 'open_interest', 'mark_iv']
    ].to_dict('records')
    
    # Max pain calculation (simplified)
    max_pain_strike = atm_options.loc[atm_options['open_interest'].idxmax()]['strike'] if not atm_options.empty else None
    
    summary = {
        "total_options": len(options_df),
        "atm_options": len(atm_options),
        "put_call_ratio_volume": round(pcr_volume, 2),
        "put_call_ratio_oi": round(pcr_oi, 2),
        "total_call_volume": round(call_vol, 2),
        "total_put_volume": round(put_vol, 2),
        "total_call_oi": round(call_oi, 2),
        "total_put_oi": round(put_oi, 2),
        "max_pain_strike": max_pain_strike,
        "top_call_strikes": top_calls,
        "top_put_strikes": top_puts,
        "market_sentiment": "BEARISH" if pcr_volume > 1.2 else "BULLISH" if pcr_volume < 0.8 else "NEUTRAL"
    }
    
    return summary

# ==================== GPT-4O MINI ANALYSIS ====================
async def analyze_chart_only(chart_file, analysis_data, session):
    """Analyze chart without options data (for Binance coins)"""
    try:
        with open(chart_file, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
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
Analyze the 4-month chart and provide:

1. **Candlestick Patterns:** Key patterns (Doji, Hammer, Engulfing, etc.)
2. **Chart Patterns:** Head & Shoulders, Triangles, Flags, Wedges
3. **Price Action:** Trend, momentum, breakouts
4. **Support & Resistance:** Key levels with exact prices
5. **Trend Lines:** Major trends and channels

**TRADING SIGNALS:**
- **SHORT TERM (1-7 days):** Entry, SL, TP, Position Size
- **MEDIUM TERM (1-4 weeks):** Entry, SL, TP, Position Size

**CONFIDENCE:** Rate 0-100% (Only provide trades if ≥70%)

**OUTPUT (JSON):**
{{
  "confidence_score": 85,
  "trend": "BULLISH/BEARISH/NEUTRAL",
  "candlestick_patterns": ["pattern1"],
  "chart_patterns": ["pattern1"],
  "support_levels": [12.50],
  "resistance_levels": [14.20],
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
  "reasoning": "Detailed analysis"
}}"""

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
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
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
            
            if "```json" in ai_response:
                json_str = ai_response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = ai_response
            
            analysis_result = json.loads(json_str)
            print(f"✅ Chart analysis completed for {analysis_data['symbol']}")
            return analysis_result
            
    except Exception as e:
        print(f"❌ Chart analysis error: {e}")
        return None

async def analyze_chart_with_options(chart_file, analysis_data, options_summary, session):
    """Analyze chart WITH options data (for BTC/ETH only)"""
    try:
        with open(chart_file, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = f"""You are an expert crypto trader analyzing {analysis_data['symbol']} with OPTIONS DATA.

**CURRENT DATA:**
- Price: ${analysis_data['current_price']:,.2f}
- 24h Change: {analysis_data['price_change_24h']:.2f}%
- Volume (24h): ${analysis_data['volume_24h']:,.0f}

**TECHNICAL INDICATORS:**
{json.dumps(analysis_data['technical_indicators'], indent=2)}

**OPTIONS CHAIN DATA (CRITICAL FOR ANALYSIS):**
{json.dumps(options_summary, indent=2)}

**OPTIONS INSIGHTS TO CONSIDER:**
1. **Put/Call Ratio:** {options_summary['put_call_ratio_volume']} (Vol) | {options_summary['put_call_ratio_oi']} (OI)
2. **Market Sentiment:** {options_summary['market_sentiment']}
3. **Max Pain Strike:** ${options_summary['max_pain_strike']}
4. **Top Call Strikes:** Heavy resistance areas
5. **Top Put Strikes:** Strong support zones

**YOUR TASK:**
Combine CHART ANALYSIS + OPTIONS FLOW to provide:

1. **Options Flow Sentiment:** Are whales bullish or bearish?
2. **Key Strike Levels:** Support/Resistance from options data
3. **Gamma Squeeze Potential:** Risk of rapid price movement
4. **Institutional Positioning:** What are big players doing?
5. **Chart Patterns:** Technical analysis
6. **Combined Strategy:** How chart + options align

**TRADING SIGNALS (considering both chart & options):**
- **SHORT TERM (1-7 days)**
- **MEDIUM TERM (1-4 weeks)**

**CONFIDENCE:** Rate 0-100%

**OUTPUT (JSON):**
{{
  "confidence_score": 85,
  "trend": "BULLISH/BEARISH/NEUTRAL",
  "options_sentiment": "BULLISH/BEARISH/NEUTRAL",
  "candlestick_patterns": ["pattern1"],
  "chart_patterns": ["pattern1"],
  "support_levels": [50000, 48000],
  "resistance_levels": [55000, 58000],
  "options_key_strikes": {{
    "support": [48000, 47000],
    "resistance": [55000, 56000]
  }},
  "gamma_squeeze_risk": "HIGH/MEDIUM/LOW",
  "institutional_flow": "BULLISH/BEARISH/NEUTRAL",
  "short_term_trade": {{
    "signal": "LONG/SHORT/NONE",
    "entry": 52000,
    "stop_loss": 50000,
    "take_profit": 56000,
    "risk_reward": 2.0,
    "position_size": "3-5%"
  }},
  "medium_term_trade": {{
    "signal": "LONG/SHORT/NONE",
    "entry": 51000,
    "stop_loss": 48000,
    "take_profit": 60000,
    "risk_reward": 3.0,
    "position_size": "5-10%"
  }},
  "reasoning": "Detailed combined analysis"
}}"""

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
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                }
            ],
            "max_tokens": 2500,
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
            
            if "```json" in ai_response:
                json_str = ai_response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = ai_response
            
            analysis_result = json.loads(json_str)
            print(f"✅ Chart + Options analysis completed for {analysis_data['symbol']}")
            return analysis_result
            
    except Exception as e:
        print(f"❌ Chart + Options analysis error: {e}")
        return None

# ==================== TELEGRAM ALERTS ====================
async def send_trade_alert(bot, symbol, analysis_result, chart_file, has_options=False):
    """Send trading alert"""
    try:
        confidence = analysis_result.get('confidence_score', 0)
        
        if confidence < ANALYSIS_ACCURACY_THRESHOLD:
            print(f"⚠️ {symbol}: Confidence {confidence}% < {ANALYSIS_ACCURACY_THRESHOLD}%, skipping")
            return
        
        with open(chart_file, 'rb') as photo:
            analysis_type = "Chart + Options" if has_options else "Chart Only"
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo,
                caption=f"🤖 **{symbol} AI Signal**\n📊 {analysis_type} | Confidence: {confidence}%"
            )
        
        msg = f"🎯 **{symbol} TRADING SIGNAL**\
        msg = f"🎯 **{symbol} TRADING SIGNAL**\n\n"
        msg += f"🔮 **Confidence:** {confidence}%\n"
        msg += f"📈 **Trend:** {analysis_result.get('trend', 'N/A')}\n"
        
        # Options-specific data (only for BTC/ETH)
        if has_options:
            msg += f"🎲 **Options Sentiment:** {analysis_result.get('options_sentiment', 'N/A')}\n"
            msg += f"⚡ **Gamma Risk:** {analysis_result.get('gamma_squeeze_risk', 'N/A')}\n"
            msg += f"🏦 **Institutional Flow:** {analysis_result.get('institutional_flow', 'N/A')}\n"
        
        msg += "\n"
        
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
        
        # Options key strikes (only for BTC/ETH)
        if has_options and analysis_result.get('options_key_strikes'):
            strikes = analysis_result['options_key_strikes']
            if strikes.get('support'):
                msg += f"🎯 **Options Support Strikes:**\n"
                for strike in strikes['support']:
                    msg += f"   • ${strike:,.0f}\n"
                msg += "\n"
            if strikes.get('resistance'):
                msg += f"🎯 **Options Resistance Strikes:**\n"
                for strike in strikes['resistance']:
                    msg += f"   • ${strike:,.0f}\n"
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
            text=f"🚀 **AI Trading Scanner Started!**\n\n🤖 Model: GPT-4o Mini Vision\n📊 Binance (Chart): {', '.join(BINANCE_COINS)}\n🎯 Deribit (Chart+Options): {', '.join(DERIBIT_OPTIONS)}\n⏰ Scan: Every {SCAN_INTERVAL//60} min\n🎯 Min Confidence: {ANALYSIS_ACCURACY_THRESHOLD}%",
            parse_mode='Markdown'
        )
    except Exception as e:
        print(f"❌ Startup error: {e}")
    
    print(f"🤖 AI Scanner Active: {len(ALL_COINS)} coins | Every {SCAN_INTERVAL//60} min\n")
    
    async with aiohttp.ClientSession() as session:
        while True:
            scan_start_time = datetime.now()
            print(f"\n{'='*70}")
            print(f"🔄 SCAN STARTED: {scan_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}\n")
            
            # ==================== SCAN BINANCE COINS (CHART ONLY) ====================
            for symbol in BINANCE_COINS:
                print(f"\n{'='*60}")
                print(f"📊 [CHART ONLY] Analyzing {symbol} at {datetime.now().strftime('%H:%M:%S')}")
                
                df = await fetch_binance_candles(session, symbol)
                ticker = await fetch_binance_ticker(session, symbol)
                funding = await fetch_binance_funding_rate(session, symbol)
                oi = await fetch_binance_open_interest(session, symbol)
                
                if df is None or df.empty:
                    print(f"⚠️ Skipping {symbol} - No data")
                    continue
                
                # Create chart
                chart_file = create_chart(df, symbol, "Binance")
                if not chart_file:
                    print(f"⚠️ Skipping {symbol} - Chart creation failed")
                    continue
                
                # Prepare analysis data
                analysis_data = prepare_analysis_data(df, ticker, funding, oi, symbol)
                
                # AI Analysis (Chart Only)
                analysis_result = await analyze_chart_only(chart_file, analysis_data, session)
                
                if analysis_result:
                    await send_trade_alert(bot, symbol, analysis_result, chart_file, has_options=False)
                
                # Cleanup
                try:
                    os.remove(chart_file)
                except:
                    pass
                
                await asyncio.sleep(10)
            
            # ==================== SCAN DERIBIT BTC/ETH (CHART + OPTIONS) ====================
            for symbol in DERIBIT_OPTIONS:
                print(f"\n{'='*60}")
                print(f"🎯 [CHART + OPTIONS] Analyzing {symbol} at {datetime.now().strftime('%H:%M:%S')}")
                
                # Fetch chart data
                df = await fetch_deribit_candles(session, symbol)
                ticker = await fetch_deribit_ticker(session, symbol)
                
                if df is None or df.empty:
                    print(f"⚠️ Skipping {symbol} - No chart data")
                    continue
                
                # Fetch options chain
                options_df = await fetch_deribit_options_chain(session, symbol)
                
                if options_df is None or options_df.empty:
                    print(f"⚠️ {symbol} - No options data, using chart only")
                    # Fallback to chart-only analysis
                    chart_file = create_chart(df, symbol, "Deribit")
                    if chart_file:
                        analysis_data = prepare_analysis_data(df, ticker, 0, 0, symbol)
                        analysis_result = await analyze_chart_only(chart_file, analysis_data, session)
                        if analysis_result:
                            await send_trade_alert(bot, symbol, analysis_result, chart_file, has_options=False)
                        try:
                            os.remove(chart_file)
                        except:
                            pass
                    continue
                
                # Create chart
                chart_file = create_chart(df, symbol, "Deribit")
                if not chart_file:
                    print(f"⚠️ Skipping {symbol} - Chart creation failed")
                    continue
                
                # Prepare analysis data
                current_price = ticker['last_price'] if ticker else df.iloc[-1]['close']
                analysis_data = prepare_analysis_data(df, ticker, 0, 0, symbol)
                
                # Prepare options summary
                options_summary = prepare_options_summary(options_df, current_price)
                
                if options_summary:
                    print(f"✅ Options summary prepared for {symbol}")
                    print(f"   Put/Call Ratio: {options_summary['put_call_ratio_volume']}")
                    print(f"   Sentiment: {options_summary['market_sentiment']}")
                    print(f"   Max Pain: ${options_summary['max_pain_strike']}")
                
                # AI Analysis (Chart + Options)
                analysis_result = await analyze_chart_with_options(
                    chart_file, analysis_data, options_summary, session
                )
                
                if analysis_result:
                    await send_trade_alert(bot, symbol, analysis_result, chart_file, has_options=True)
                
                # Cleanup
                try:
                    os.remove(chart_file)
                except:
                    pass
                
                await asyncio.sleep(10)
            
            # ==================== SCAN COMPLETE ====================
            scan_end_time = datetime.now()
            scan_duration = (scan_end_time - scan_start_time).total_seconds()
            
            print(f"\n{'='*70}")
            print(f"✅ SCAN COMPLETED: {scan_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️ Duration: {scan_duration:.1f} seconds")
            print(f"⏳ Next scan in {SCAN_INTERVAL//60} minutes...")
            print(f"{'='*70}\n")
            
            await asyncio.sleep(SCAN_INTERVAL)

# ==================== TELEGRAM COMMANDS ====================
async def start(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **AI Crypto Trading Scanner**\n\n"
        f"🧠 Model: GPT-4o Mini Vision\n"
        f"📊 Binance (Chart): {len(BINANCE_COINS)} coins\n"
        f"🎯 Deribit (Chart+Options): {len(DERIBIT_OPTIONS)} coins\n"
        f"⏰ Analysis: Every {SCAN_INTERVAL//60} min\n"
        f"🎯 Min Confidence: {ANALYSIS_ACCURACY_THRESHOLD}%\n\n"
        f"**Analysis Types:**\n"
        f"• Binance: Technical chart analysis\n"
        f"• Deribit BTC/ETH: Chart + Options flow\n\n"
        f"Commands: /start /status",
        parse_mode='Markdown'
    )

async def status(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"✅ Running\n"
        f"📊 Chart Analysis: {len(BINANCE_COINS)} coins\n"
        f"🎯 Chart+Options: {len(DERIBIT_OPTIONS)} coins\n"
        f"⏰ Scan: Every {SCAN_INTERVAL//60} min",
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
