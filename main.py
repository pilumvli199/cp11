import os
import time
import base64
from io import BytesIO
from datetime import datetime
import requests
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from apscheduler.schedulers.background import BackgroundScheduler
from openai import OpenAI
from flask import Flask, render_template_string, jsonify
import telebot

# Initialize
app = Flask(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Coins to track
COINS = [
    'BTC', 'ETH', 'USDT', 'BNB', 'XRP', 
    'SOL', 'USDC', 'TRX', 'DOGE', 'ADA',
    'LINK', 'BCH', 'XLM', 'SUI', 'AVAX'
]

TIMEFRAMES = ['1h', '4h', '1d']

# Store latest signals
latest_signals = {}

def fetch_candlestick_data(symbol, timeframe='1h', limit=2800):
    """Fetch candlestick data from Binance API (free, no auth needed)"""
    try:
        # Convert timeframe for Binance
        tf_map = {'1h': '1h', '4h': '4h', '1d': '1d'}
        interval = tf_map.get(timeframe, '1h')
        
        url = f'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': f'{symbol}USDT',
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Clean and convert
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    except Exception as e:
        print(f"Error fetching {symbol} data: {e}")
        return None

def calculate_swing_points(df, window=5):
    """Identify swing highs and lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(df) - window):
        # Swing High
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
            swing_highs.append((df.index[i], df['high'].iloc[i]))
        
        # Swing Low
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
            swing_lows.append((df.index[i], df['low'].iloc[i]))
    
    return swing_highs, swing_lows

def identify_support_resistance(df, num_levels=3):
    """Find key support and resistance levels"""
    swing_highs, swing_lows = calculate_swing_points(df)
    
    # Get recent swing points (last 50)
    recent_highs = [price for _, price in swing_highs[-50:]]
    recent_lows = [price for _, price in swing_lows[-50:]]
    
    # Cluster nearby levels
    resistance_levels = []
    support_levels = []
    
    if recent_highs:
        resistance_levels = sorted(set(recent_highs), reverse=True)[:num_levels]
    
    if recent_lows:
        support_levels = sorted(set(recent_lows))[:num_levels]
    
    return support_levels, resistance_levels

def draw_chart(df, symbol, timeframe, support_levels, resistance_levels):
    """Generate candlestick chart with S/R levels"""
    
    # Take last 200 candles for chart
    df_chart = df.tail(200).copy()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot candlesticks
    mpf.plot(df_chart, type='candle', style='charles', ax=ax, 
             volume=False, ylabel='Price', warn_too_much_data=9999)
    
    # Add support levels (green)
    for level in support_levels:
        ax.axhline(y=level, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Support: {level:.2f}')
    
    # Add resistance levels (red)
    for level in resistance_levels:
        ax.axhline(y=level, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Resistance: {level:.2f}')
    
    ax.set_title(f'{symbol} - {timeframe} Timeframe | Pure Price Action Analysis', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Save to bytes
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def analyze_with_gpt(chart_image, candlestick_data, symbol, timeframe, support_levels, resistance_levels):
    """Send chart and data to GPT-4o Mini for analysis"""
    
    # Encode image
    img_base64 = base64.b64encode(chart_image.read()).decode('utf-8')
    
    # Prepare recent candles text
    recent_candles = candlestick_data.tail(20).to_dict('records')
    candles_text = "\n".join([
        f"Time: {i}, O:{c['open']:.2f}, H:{c['high']:.2f}, L:{c['low']:.2f}, C:{c['close']:.2f}"
        for i, c in enumerate(recent_candles)
    ])
    
    prompt = f"""You are a professional pure price action trader analyzing {symbol} on {timeframe} timeframe.

**CHART DATA:**
- Current Price: {candlestick_data['close'].iloc[-1]:.2f}
- Support Levels: {support_levels}
- Resistance Levels: {resistance_levels}

**RECENT CANDLESTICKS (Last 20):**
{candles_text}

**TRADING RULES - STRICT PRICE ACTION ONLY:**

1. **CANDLESTICK PATTERNS (Reversal Signals):**
   - Bullish: Hammer, Bullish Engulfing, Morning Star, Piercing Pattern
   - Bearish: Shooting Star, Bearish Engulfing, Evening Star, Dark Cloud Cover

2. **CHART PATTERNS:**
   - Reversal: Head & Shoulders, Double Top/Bottom, V-Reversal
   - Continuation: Flags, Triangles, Channels

3. **MARKET STRUCTURE:**
   - Uptrend: Higher Highs + Higher Lows
   - Downtrend: Lower Highs + Lower Lows
   - Range: Bouncing between S/R

4. **ENTRY RULES:**
   - LONG: Bullish reversal pattern AT support + rejection wick
   - SHORT: Bearish reversal pattern AT resistance + rejection wick
   - Avoid choppy/ranging markets without clear setup

5. **EXIT RULES:**
   - Take profit at next S/R level
   - Exit if opposite pattern forms
   - Trail stop below recent swing low (long) or above swing high (short)

**YOUR TASK:**
Analyze the chart image and recent price action. Give:
1. Market Structure (trend/range)
2. Key Pattern Identified (if any)
3. Trade Signal: LONG / SHORT / NO TRADE
4. Entry Price & Reason
5. Stop Loss Level
6. Take Profit Target
7. Risk/Reward Ratio

Be conservative. Only signal high-probability setups. If no clear setup, say NO TRADE.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        analysis = response.choices[0].message.content
        return analysis
    
    except Exception as e:
        return f"GPT Analysis Error: {e}"

def scan_coin(symbol, timeframe):
    """Complete scan workflow for one coin"""
    print(f"\n🔍 Scanning {symbol} on {timeframe}...")
    
    # 1. Fetch data
    df = fetch_candlestick_data(symbol, timeframe)
    if df is None or len(df) < 100:
        return {"error": "Data fetch failed"}
    
    # 2. Calculate S/R levels
    support_levels, resistance_levels = identify_support_resistance(df)
    
    # 3. Generate chart
    chart_img = draw_chart(df, symbol, timeframe, support_levels, resistance_levels)
    
    # 4. GPT Analysis
    chart_img_copy = BytesIO(chart_img.getvalue())
    analysis = analyze_with_gpt(chart_img_copy, df, symbol, timeframe, support_levels, resistance_levels)
    
    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": float(df['close'].iloc[-1]),
        "support": support_levels,
        "resistance": resistance_levels,
        "analysis": analysis,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"✅ {symbol} scan complete!")
    return result

def scan_all_coins():
    """Scan all coins on all timeframes"""
    print("\n" + "="*50)
    print(f"🚀 Starting Full Scan - {datetime.now()}")
    print("="*50)
    
    results = {}
    
    for coin in COINS:
        for tf in TIMEFRAMES:
            key = f"{coin}_{tf}"
            try:
                result = scan_coin(coin, tf)
                results[key] = result
                latest_signals[key] = result
                time.sleep(2)  # Rate limit protection
            except Exception as e:
                print(f"❌ Error scanning {key}: {e}")
                results[key] = {"error": str(e)}
    
    print("\n✅ Full scan completed!")
    return results

# Flask Routes
@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pure Price Action Bot</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #1a1a1a; color: #fff; }
            h1 { color: #4CAF50; }
            .signal { background: #2d2d2d; padding: 15px; margin: 10px 0; border-radius: 8px; }
            .long { border-left: 4px solid #4CAF50; }
            .short { border-left: 4px solid #f44336; }
            .no-trade { border-left: 4px solid #888; }
            pre { background: #000; padding: 10px; overflow-x: auto; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; 
                     border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #45a049; }
        </style>
    </head>
    <body>
        <h1>🤖 Pure Price Action Trading Bot</h1>
        <p>Scanning: """ + ", ".join(COINS) + """</p>
        <p>Timeframes: """ + ", ".join(TIMEFRAMES) + """</p>
        <button onclick="location.reload()">🔄 Refresh Signals</button>
        <button onclick="manualScan()">▶️ Manual Scan</button>
        <hr>
        <div id="signals"></div>
        
        <script>
            async function loadSignals() {
                const res = await fetch('/signals');
                const data = await res.json();
                const div = document.getElementById('signals');
                div.innerHTML = '';
                
                for (const [key, signal] of Object.entries(data)) {
                    if (signal.error) continue;
                    
                    let signalClass = 'no-trade';
                    if (signal.analysis.includes('LONG')) signalClass = 'long';
                    else if (signal.analysis.includes('SHORT')) signalClass = 'short';
                    
                    div.innerHTML += `
                        <div class="signal ${signalClass}">
                            <h3>${signal.symbol} (${signal.timeframe}) - $${signal.current_price}</h3>
                            <p><small>${signal.timestamp}</small></p>
                            <pre>${signal.analysis}</pre>
                        </div>
                    `;
                }
            }
            
            async function manualScan() {
                alert('Starting manual scan... This will take a few minutes.');
                await fetch('/manual-scan');
                alert('Scan complete! Refreshing...');
                location.reload();
            }
            
            loadSignals();
            setInterval(loadSignals, 60000); // Refresh every minute
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/signals')
def get_signals():
    return jsonify(latest_signals)

@app.route('/manual-scan')
def manual_scan():
    scan_all_coins()
    return jsonify({"status": "complete"})

if __name__ == '__main__':
    # Schedule hourly scans
    scheduler = BackgroundScheduler()
    scheduler.add_job(scan_all_coins, 'interval', hours=1)
    scheduler.start()
    
    print("🚀 Bot started! Running initial scan...")
    scan_all_coins()
    
    # Start Flask server
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
