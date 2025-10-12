import os
import time
import json
from io import BytesIO
from datetime import datetime
import requests
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from apscheduler.schedulers.background import BackgroundScheduler
from openai import OpenAI
from flask import Flask, render_template_string, jsonify
import telebot

# Initialize
app = Flask(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Telegram Bot
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

# Coins to track
COINS = [
    'BTC', 'ETH', 'USDT', 'BNB', 'XRP', 
    'SOL', 'USDC', 'TRX', 'DOGE', 'ADA',
    'LINK', 'BCH', 'XLM', 'SUI', 'AVAX'
]

TIMEFRAMES = ['1h', '4h', '1d']

# Store latest signals
latest_signals = {}

def send_startup_message():
    """Send bot startup notification to Telegram"""
    if not bot or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram not configured - skipping startup message")
        return
    
    try:
        startup_msg = f"""
🤖 **BOT STARTED SUCCESSFULLY!**

✅ Pure Price Action Trading Bot is now LIVE!

🧠 **Powered by:** GPT-4o Mini (Vision + Analysis)

📊 **Monitoring:**
• 14 Cryptocurrencies
• 3 Timeframes (1h, 4h, 1d)
• Total: 42 scans per cycle

⏰ **Scan Frequency:** Every 1 hour
🎯 **Signal Types:** LONG / SHORT / NO TRADE

🔔 You will receive alerts only for:
🟢 LONG signals
🔴 SHORT signals

⚡ First scan starting now...

Bot Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=startup_msg,
            parse_mode='Markdown'
        )
        print("✅ Startup message sent to Telegram!")
        
    except Exception as e:
        print(f"❌ Failed to send startup message: {e}")

def fetch_candlestick_data(symbol, timeframe='1h', limit=2800):
    """Fetch candlestick data from Binance API"""
    try:
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
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    except Exception as e:
        print(f"❌ Error fetching {symbol} data: {e}")
        return None

def calculate_swing_points(df, window=5):
    """Identify swing highs and lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(df) - window):
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
            swing_highs.append({'index': i, 'price': df['high'].iloc[i], 'time': df.index[i]})
        
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
            swing_lows.append({'index': i, 'price': df['low'].iloc[i], 'time': df.index[i]})
    
    return swing_highs, swing_lows

def identify_support_resistance(df, num_levels=3):
    """Find key support and resistance levels"""
    swing_highs, swing_lows = calculate_swing_points(df)
    
    recent_highs = [s['price'] for s in swing_highs[-50:]]
    recent_lows = [s['price'] for s in swing_lows[-50:]]
    
    resistance_levels = []
    support_levels = []
    
    if recent_highs:
        resistance_levels = sorted(set(recent_highs), reverse=True)[:num_levels]
    
    if recent_lows:
        support_levels = sorted(set(recent_lows))[:num_levels]
    
    return support_levels, resistance_levels, swing_highs, swing_lows

def analyze_with_gpt(candlestick_data, symbol, timeframe, support_levels, resistance_levels, swing_highs, swing_lows):
    """Send RAW candlestick data to GPT-4o Mini for deep analysis"""
    
    print(f"🧠 Sending data to GPT-4o Mini for analysis...")
    
    # Prepare last 100 candles as JSON
    recent_data = candlestick_data.tail(100).reset_index()
    candles_json = recent_data.to_dict('records')
    
    # Prepare swing points
    recent_swing_highs = [{'price': s['price'], 'time': str(s['time'])} for s in swing_highs[-20:]]
    recent_swing_lows = [{'price': s['price'], 'time': str(s['time'])} for s in swing_lows[-20:]]
    
    # Build comprehensive prompt
    prompt = f"""You are an expert pure price action trader. Analyze {symbol} on {timeframe} timeframe.

**CURRENT MARKET DATA:**
- Current Price: ${candlestick_data['close'].iloc[-1]:.2f}
- Support Levels: {support_levels}
- Resistance Levels: {resistance_levels}

**RAW CANDLESTICK DATA (Last 100 Candles in JSON):**
{json.dumps(candles_json[-50:], indent=2, default=str)}

**SWING ANALYSIS:**
- Recent Swing Highs: {recent_swing_highs[-5:]}
- Recent Swing Lows: {recent_swing_lows[-5:]}

**YOUR TASK - DEEP PRICE ACTION ANALYSIS:**

1. **IDENTIFY CHART PATTERNS:**
   - Head & Shoulders (regular/inverse)
   - Double/Triple Top/Bottom
   - Ascending/Descending Triangle
   - Symmetrical Triangle
   - Flag/Pennant
   - Wedge (rising/falling)
   - Channel (ascending/descending/horizontal)

2. **IDENTIFY CANDLESTICK PATTERNS (Last 5-10 candles):**
   - Bullish: Hammer, Bullish Engulfing, Morning Star, Piercing Pattern, Three White Soldiers
   - Bearish: Shooting Star, Bearish Engulfing, Evening Star, Dark Cloud Cover, Three Black Crows
   - Indecision: Doji, Spinning Top

3. **DRAW TRENDLINES:**
   - Connect swing highs for resistance trendline
   - Connect swing lows for support trendline
   - Identify trendline breaks or bounces

4. **MARKET STRUCTURE:**
   - Higher Highs + Higher Lows = Uptrend
   - Lower Highs + Lower Lows = Downtrend
   - Ranging = Choppy/Sideways

5. **GENERATE TRADE SIGNAL:**
   - LONG: Bullish pattern + support bounce + uptrend
   - SHORT: Bearish pattern + resistance rejection + downtrend
   - NO TRADE: No clear setup, choppy, conflicting signals

**STRICT OUTPUT FORMAT (MUST FOLLOW EXACTLY):**

SIGNAL: [LONG/SHORT/NO TRADE]
CHART_PATTERN: [Pattern name or "None"]
CANDLESTICK_PATTERN: [Pattern name or "None"]
TRENDLINE: [Uptrend/Downtrend/Range/Break]
MARKET_STRUCTURE: [Higher Highs Higher Lows / Lower Highs Lower Lows / Range]
ENTRY: $[price]
STOP_LOSS: $[price]
TAKE_PROFIT: $[price]
RISK_REWARD: [ratio]
REASON: [2-3 sentence explanation focusing on price action]

**IMPORTANT:**
- Be conservative - only signal high-probability setups
- Price action ONLY - no indicators
- If unclear, say NO TRADE
"""

    try:
        print("⏳ Waiting for GPT-4o Mini response...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert pure price action trader. Analyze raw candlestick data and identify patterns, trendlines, and trade setups."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        analysis = response.choices[0].message.content
        print(f"✅ GPT-4o Mini analysis received!")
        print(f"📊 Analysis Preview:\n{analysis[:200]}...\n")
        
        return analysis
    
    except Exception as e:
        print(f"❌ GPT Analysis Error: {e}")
        return f"GPT Analysis Error: {e}"

def parse_gpt_analysis(analysis):
    """Extract key info from GPT analysis"""
    lines = analysis.split('\n')
    parsed = {
        'signal': 'NO TRADE',
        'chart_pattern': 'None',
        'candlestick_pattern': 'None',
        'trendline': 'Unknown',
        'market_structure': 'Unknown'
    }
    
    for line in lines:
        line_upper = line.upper()
        if line_upper.startswith('SIGNAL:'):
            if 'LONG' in line_upper:
                parsed['signal'] = 'LONG'
            elif 'SHORT' in line_upper:
                parsed['signal'] = 'SHORT'
        elif line_upper.startswith('CHART_PATTERN:'):
            parsed['chart_pattern'] = line.split(':', 1)[1].strip()
        elif line_upper.startswith('CANDLESTICK_PATTERN:'):
            parsed['candlestick_pattern'] = line.split(':', 1)[1].strip()
        elif line_upper.startswith('TRENDLINE:'):
            parsed['trendline'] = line.split(':', 1)[1].strip()
        elif line_upper.startswith('MARKET_STRUCTURE:'):
            parsed['market_structure'] = line.split(':', 1)[1].strip()
    
    return parsed

def draw_enhanced_chart(df, symbol, timeframe, support_levels, resistance_levels, gpt_data):
    """Generate chart with GPT analysis annotations"""
    
    df_chart = df.tail(200).copy()
    
    fig, ax = plt.subplots(figsize=(18, 11))
    
    # Plot candlesticks
    mpf.plot(df_chart, type='candle', style='charles', ax=ax, 
             volume=False, ylabel='Price', warn_too_much_data=9999)
    
    # Support levels
    for level in support_levels[:3]:
        ax.axhline(y=level, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(len(df_chart)-5, level, f'Support: ${level:.2f}', 
                fontsize=9, color='green', va='bottom', fontweight='bold')
    
    # Resistance levels
    for level in resistance_levels[:3]:
        ax.axhline(y=level, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(len(df_chart)-5, level, f'Resistance: ${level:.2f}', 
                fontsize=9, color='red', va='top', fontweight='bold')
    
    # Add GPT analysis annotations
    title_color = '#4CAF50' if gpt_data['signal'] == 'LONG' else '#f44336' if gpt_data['signal'] == 'SHORT' else '#888'
    
    title = f"{symbol} - {timeframe} | GPT-4o Mini Analysis\n"
    title += f"Signal: {gpt_data['signal']} | Chart Pattern: {gpt_data['chart_pattern']}\n"
    title += f"Candlestick: {gpt_data['candlestick_pattern']} | Trend: {gpt_data['trendline']}"
    
    ax.set_title(title, fontsize=14, fontweight='bold', color=title_color, pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add signal marker on last candle
    if gpt_data['signal'] in ['LONG', 'SHORT']:
        marker_color = 'green' if gpt_data['signal'] == 'LONG' else 'red'
        marker_symbol = '^' if gpt_data['signal'] == 'LONG' else 'v'
        ax.plot(len(df_chart)-1, df_chart['close'].iloc[-1], 
                marker=marker_symbol, markersize=15, color=marker_color, zorder=5)
    
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#1a1a1a')
    buf.seek(0)
    plt.close()
    
    return buf

def send_telegram_alert(symbol, timeframe, analysis, current_price, gpt_data, chart_img=None):
    """Send trading signal to Telegram with GPT insights"""
    if not bot or not TELEGRAM_CHAT_ID:
        return
    
    try:
        signal_type = gpt_data['signal']
        
        if signal_type in ["LONG", "SHORT"]:
            emoji = "🟢" if signal_type == "LONG" else "🔴"
            
            message = f"""
{emoji} **{signal_type} SIGNAL DETECTED!**

💰 **{symbol}** ({timeframe})
💵 Current Price: ${current_price:.2f}

🧠 **GPT-4o Mini Analysis:**

📊 **Chart Pattern:** {gpt_data['chart_pattern']}
🕯️ **Candlestick Pattern:** {gpt_data['candlestick_pattern']}
📈 **Trendline:** {gpt_data['trendline']}
🏗️ **Market Structure:** {gpt_data['market_structure']}

**Full Analysis:**
{analysis}

⏰ {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            """
            
            if chart_img:
                chart_img.seek(0)
                bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=chart_img,
                    caption=message,
                    parse_mode='Markdown'
                )
                print(f"📲 Telegram alert sent: {symbol} {timeframe} - {signal_type}")
        
    except Exception as e:
        print(f"❌ Telegram send error: {e}")

def send_scan_summary(signal_count, signal_details):
    """Send scan cycle summary to Telegram"""
    if not bot or not TELEGRAM_CHAT_ID:
        return
    
    try:
        long_list = "\n".join([f"  • {s}" for s in signal_details["LONG"]]) if signal_details["LONG"] else "  • None"
        short_list = "\n".join([f"  • {s}" for s in signal_details["SHORT"]]) if signal_details["SHORT"] else "  • None"
        
        summary_msg = f"""
📊 **SCAN CYCLE COMPLETED**

━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 **LONG Signals: {signal_count['LONG']}**
{long_list}

🔴 **SHORT Signals: {signal_count['SHORT']}**
{short_list}

⚪ **NO TRADE: {signal_count['NO TRADE']}**

━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Total Scans: {sum(signal_count.values())}
⏰ Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
⏱️ Next Scan: 1 hour

🧠 Powered by: GPT-4o Mini
🤖 Bot Status: Active
        """
        
        bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=summary_msg,
            parse_mode='Markdown'
        )
        print("📲 Scan summary sent to Telegram!")
        
    except Exception as e:
        print(f"❌ Failed to send scan summary: {e}")

def scan_coin(symbol, timeframe):
    """Complete scan workflow for one coin"""
    print(f"\n🔍 Scanning {symbol} on {timeframe}...")
    
    # 1. Fetch data
    df = fetch_candlestick_data(symbol, timeframe)
    if df is None or len(df) < 100:
        print(f"❌ {symbol} {timeframe}: Data fetch failed")
        return {"error": "Data fetch failed"}
    
    # 2. Calculate S/R levels and swing points
    support_levels, resistance_levels, swing_highs, swing_lows = identify_support_resistance(df)
    print(f"📊 {symbol} {timeframe}: Support={[f'${s:.2f}' for s in support_levels[:2]]}, Resistance={[f'${r:.2f}' for r in resistance_levels[:2]]}")
    
    # 3. GPT-4o Mini Deep Analysis
    analysis = analyze_with_gpt(df, symbol, timeframe, support_levels, resistance_levels, swing_highs, swing_lows)
    
    # 4. Parse GPT response
    gpt_data = parse_gpt_analysis(analysis)
    
    # Log the result
    if gpt_data['signal'] == "NO TRADE":
        print(f"⚪ {symbol} {timeframe}: NO TRADE - No clear setup found")
    else:
        print(f"{'🟢' if gpt_data['signal'] == 'LONG' else '🔴'} {symbol} {timeframe}: **{gpt_data['signal']} SIGNAL CONFIRMED!**")
        print(f"   📊 Chart Pattern: {gpt_data['chart_pattern']}")
        print(f"   🕯️ Candlestick: {gpt_data['candlestick_pattern']}")
        print(f"   📈 Trend: {gpt_data['trendline']}")
        
        # Extract trade details
        for line in analysis.split('\n'):
            if any(keyword in line.upper() for keyword in ['ENTRY', 'STOP', 'TAKE', 'RISK']):
                print(f"   {line.strip()}")
    
    # 5. Generate enhanced chart
    chart_img = draw_enhanced_chart(df, symbol, timeframe, support_levels, resistance_levels, gpt_data)
    
    # 6. Send Telegram Alert
    if gpt_data['signal'] in ["LONG", "SHORT"]:
        chart_img_copy = BytesIO(chart_img.getvalue())
        send_telegram_alert(symbol, timeframe, analysis, df['close'].iloc[-1], gpt_data, chart_img_copy)
    
    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": float(df['close'].iloc[-1]),
        "support": support_levels,
        "resistance": resistance_levels,
        "analysis": analysis,
        "signal_type": gpt_data['signal'],
        "chart_pattern": gpt_data['chart_pattern'],
        "candlestick_pattern": gpt_data['candlestick_pattern'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"✅ {symbol} {timeframe} scan complete!\n")
    return result

def scan_all_coins():
    """Scan all coins on all timeframes"""
    print("\n" + "="*80)
    print(f"🚀 STARTING FULL SCAN CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"📊 Total Scans: {len(COINS)} coins × {len(TIMEFRAMES)} timeframes = {len(COINS) * len(TIMEFRAMES)} scans")
    print(f"🧠 Using: GPT-4o Mini for deep price action analysis")
    print("="*80 + "\n")
    
    results = {}
    signal_count = {"LONG": 0, "SHORT": 0, "NO TRADE": 0}
    signal_details = {"LONG": [], "SHORT": []}
    
    for coin in COINS:
        for tf in TIMEFRAMES:
            key = f"{coin}_{tf}"
            try:
                result = scan_coin(coin, tf)
                results[key] = result
                latest_signals[key] = result
                
                if "signal_type" in result:
                    signal_count[result["signal_type"]] += 1
                    
                    if result["signal_type"] in ["LONG", "SHORT"]:
                        signal_details[result["signal_type"]].append(f"{coin} ({tf})")
                
                time.sleep(2)
            except Exception as e:
                print(f"❌ Error scanning {key}: {e}")
                results[key] = {"error": str(e)}
    
    print("\n" + "="*80)
    print("📈 SCAN CYCLE COMPLETE - SUMMARY")
    print("="*80)
    print(f"🟢 LONG Signals:     {signal_count['LONG']}")
    print(f"🔴 SHORT Signals:    {signal_count['SHORT']}")
    print(f"⚪ NO TRADE:         {signal_count['NO TRADE']}")
    print(f"✅ Total Processed:  {sum(signal_count.values())}")
    print(f"⏰ Completed At:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Next Scan In:     1 hour")
    print("="*80 + "\n")
    
    send_scan_summary(signal_count, signal_details)
    
    return results

# Flask Routes
@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pure Price Action Bot - GPT-4o Mini</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #1a1a1a; color: #fff; }
            h1 { color: #4CAF50; }
            .signal { background: #2d2d2d; padding: 15px; margin: 10px 0; border-radius: 8px; }
            .long { border-left: 4px solid #4CAF50; }
            .short { border-left: 4px solid #f44336; }
            .no-trade { border-left: 4px solid #888; opacity: 0.6; }
            pre { background: #000; padding: 10px; overflow-x: auto; font-size: 12px; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; 
                     border-radius: 5px; cursor: pointer; font-size: 16px; margin: 5px; }
            button:hover { background: #45a049; }
            .stats { display: flex; gap: 20px; margin: 20px 0; }
            .stat-box { background: #2d2d2d; padding: 15px; border-radius: 8px; flex: 1; text-align: center; }
            .stat-box h3 { margin: 0; font-size: 32px; }
            .stat-box p { margin: 5px 0; color: #888; }
            .pattern-tag { display: inline-block; background: #333; padding: 5px 10px; 
                          border-radius: 5px; margin: 5px; font-size: 11px; }
        </style>
    </head>
    <body>
        <h1>🤖 Pure Price Action Trading Bot</h1>
        <p>🧠 Powered by GPT-4o Mini AI Analysis</p>
        <p>Scanning: """ + ", ".join(COINS) + """</p>
        <p>Timeframes: """ + ", ".join(TIMEFRAMES) + """</p>
        <button onclick="location.reload()">🔄 Refresh</button>
        <button onclick="manualScan()">▶️ Manual Scan</button>
        
        <div class="stats" id="stats"></div>
        <hr>
        <div id="signals"></div>
        
        <script>
            async function loadSignals() {
                const res = await fetch('/signals');
                const data = await res.json();
                const div = document.getElementById('signals');
                const statsDiv = document.getElementById('stats');
                
                let longCount = 0, shortCount = 0, noTradeCount = 0;
                div.innerHTML = '';
                
                for (const [key, signal] of Object.entries(data)) {
                    if (signal.error) continue;
                    
                    let signalClass = 'no-trade';
                    if (signal.signal_type === 'LONG') { signalClass = 'long'; longCount++; }
                    else if (signal.signal_type === 'SHORT') { signalClass = 'short'; shortCount++; }
                    else { noTradeCount++; }
                    
                    let patterns = '';
                    if (signal.chart_pattern && signal.chart_pattern !== 'None') {
                        patterns += `<span class="pattern-tag">📊 ${signal.chart_pattern}</span>`;
                    }
                    if (signal.candlestick_pattern && signal.candlestick_pattern !== 'None') {
                        patterns += `<span class="pattern-tag">🕯️ ${signal.candlestick_pattern}</span>`;
                    }
                    
                    div.innerHTML += `
                        <div class="signal ${signalClass}">
                            <h3>${signal.symbol} (${signal.timeframe}) - $${signal.current_price.toFixed(2)} - ${signal.signal_type}</h3>
                            <div>${patterns}</div>
                            <p><small>${signal.timestamp}</small></p>
                            <pre>${signal.analysis}</pre>
                        </div>
                    `;
                }
                
                statsDiv.innerHTML = `
                    <div class="stat-box" style="border-left: 4px solid #4CAF50;">
                        <h3>🟢 ${longCount}</h3>
                        <p>LONG Signals</p>
                    </div>
                    <div class="stat-box" style="border-left: 4px solid #f44336;">
                        <h3>🔴 ${shortCount}</h3>
                        <p>SHORT Signals</p>
                    </div>
                    <div class="stat-box" style="border-left: 4px solid #888;">
                        <h3>⚪ ${noTradeCount}</h3>
                        <p>NO TRADE</p>
                    </div>
                `;
            }
            
            async function manualScan() {
                if (!confirm('Start manual scan? This will take 2-3 minutes.')) return;
                document.body.style.opacity = '0.5';
                await fetch('/manual-scan');
                document.body.style.opacity = '1';
                alert('Scan complete!');
                location.reload();
            }
            
            loadSignals();
            setInterval(loadSignals, 60000);
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
    print("\n" + "="*80)
    print("🤖 PURE PRICE ACTION TRADING BOT")
    print("🧠 POWERED BY GPT-4o MINI AI ANALYSIS")
    print("="*80)
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Monitoring: {len(COINS)} coins × {len(TIMEFRAMES)} timeframes = {len(COINS) * len(TIMEFRAMES)} scans")
    print(f"🔄 Scan Frequency: Every 1 hour")
    print(f"📱 Telegram: {'✅ Configured' if bot else '❌ Not configured'}")
    print("="*80 + "\n")
    
    send_startup_message()
    
    scheduler = BackgroundScheduler()
    scheduler.add_job(scan_all_coins, 'interval', hours=1)
    scheduler.start()
    
    print("🚀 Running initial scan...\n")
    scan_all_coins()
    
    port = int(os.getenv('PORT', 5000))
    print(f"\n🌐 Starting Flask server on port {port}...\n")
    app.run(host='0.0.0.0', port=port, debug=False)
