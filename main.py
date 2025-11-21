#!/usr/bin/env python3
"""
DERIBIT & BINANCE OPTIONS DASHBOARD
====================================
- Fetches option chain data from both exchanges
- Uses PUBLIC endpoints (no API key needed)
- Shows OI, Volume, IV, Greeks, etc.
"""

import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import asyncio
from io import BytesIO
from datetime import datetime
from telegram import Bot
import logging

# ==================== CONFIGURATION ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Base URLs
DERIBIT_URL = "https://www.deribit.com/api/v2"
BINANCE_URL = "https://eapi.binance.com"

plt.style.use('dark_background')


class DeribitDashboard:
    """Deribit Options Chain - Public API"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def safe_float(self, val, default=0.0):
        if val is None:
            return default
        try:
            return float(val)
        except:
            return default

    def get_spot_price(self, currency='BTC'):
        """Get BTC/ETH spot price"""
        try:
            url = f"{DERIBIT_URL}/public/get_index_price"
            params = {'index_name': f"{currency.lower()}_usd"}
            res = self.session.get(url, params=params, timeout=10)
            
            if res.status_code == 200:
                data = res.json()
                return self.safe_float(data.get('result', {}).get('index_price'))
            return 0
        except Exception as e:
            logger.error(f"Deribit Spot Error: {e}")
            return 0

    def get_expiries(self, currency='BTC'):
        """Get available expiry dates"""
        try:
            url = f"{DERIBIT_URL}/public/get_instruments"
            params = {
                'currency': currency,
                'kind': 'option',
                'expired': 'false'
            }
            res = self.session.get(url, params=params, timeout=10)
            
            if res.status_code == 200:
                instruments = res.json().get('result', [])
                expiries = {}
                now = datetime.utcnow()
                
                for inst in instruments:
                    exp_ts = inst.get('expiration_timestamp', 0) / 1000
                    exp_dt = datetime.utcfromtimestamp(exp_ts)
                    if exp_dt > now:
                        date_str = exp_dt.strftime('%d%b%y').upper()
                        expiries[date_str] = exp_dt
                
                return sorted(expiries.items(), key=lambda x: x[1])
            return []
        except Exception as e:
            logger.error(f"Deribit Expiries Error: {e}")
            return []

    def get_chain_data(self, currency='BTC'):
        """Fetch Deribit option chain"""
        try:
            # 1. Get spot
            spot = self.get_spot_price(currency)
            if spot == 0:
                return None, 0, ""
            
            logger.info(f"💰 Deribit {currency} Spot: ${spot:,.2f}")

            # 2. Get nearest expiry
            expiries = self.get_expiries(currency)
            if not expiries:
                return None, 0, ""
            
            exp_str, exp_dt = expiries[0]
            logger.info(f"📅 Deribit Using expiry: {exp_str}")

            # 3. Get instruments for this expiry
            url = f"{DERIBIT_URL}/public/get_instruments"
            params = {
                'currency': currency,
                'kind': 'option',
                'expired': 'false'
            }
            res = self.session.get(url, params=params, timeout=15)
            
            if res.status_code != 200:
                return None, 0, ""
            
            instruments = res.json().get('result', [])
            
            # Filter by expiry
            target_instruments = []
            for inst in instruments:
                exp_ts = inst.get('expiration_timestamp', 0) / 1000
                exp_date = datetime.utcfromtimestamp(exp_ts).strftime('%d%b%y').upper()
                if exp_date == exp_str:
                    target_instruments.append(inst['instrument_name'])
            
            logger.info(f"📊 Found {len(target_instruments)} instruments")

            # 4. Get ticker data for all instruments
            calls_data = {}
            puts_data = {}
            
            for inst_name in target_instruments[:50]:  # Limit to avoid rate limits
                try:
                    ticker_url = f"{DERIBIT_URL}/public/ticker"
                    ticker_res = self.session.get(ticker_url, 
                                                params={'instrument_name': inst_name},
                                                timeout=5)
                    
                    if ticker_res.status_code != 200:
                        continue
                    
                    ticker = ticker_res.json().get('result', {})
                    
                    # Parse instrument name: BTC-27DEC24-95000-C
                    parts = inst_name.split('-')
                    if len(parts) != 4:
                        continue
                    
                    strike = self.safe_float(parts[2])
                    opt_type = parts[3]  # C or P
                    
                    # Extract data
                    data = {
                        'ltp': self.safe_float(ticker.get('mark_price')),
                        'oi': self.safe_float(ticker.get('open_interest')),  # In contracts
                        'vol': self.safe_float(ticker.get('stats', {}).get('volume_usd')),
                        'iv': self.safe_float(ticker.get('mark_iv')) * 100,  # Convert to %
                        'delta': self.safe_float(ticker.get('greeks', {}).get('delta')),
                        'gamma': self.safe_float(ticker.get('greeks', {}).get('gamma')),
                        'theta': self.safe_float(ticker.get('greeks', {}).get('theta')),
                        'vega': self.safe_float(ticker.get('greeks', {}).get('vega'))
                    }
                    
                    if opt_type == 'C':
                        calls_data[strike] = data
                    else:
                        puts_data[strike] = data
                    
                    time.sleep(0.05)  # Rate limit protection
                    
                except Exception as e:
                    continue
            
            logger.info(f"✅ Parsed: {len(calls_data)} calls, {len(puts_data)} puts")
            
            # 5. Build DataFrame
            all_strikes = sorted(set(list(calls_data.keys()) + list(puts_data.keys())))
            
            rows = []
            for strike in all_strikes:
                c = calls_data.get(strike, {})
                p = puts_data.get(strike, {})
                
                rows.append({
                    'strike': strike,
                    'c_vol': c.get('vol', 0),
                    'c_oi': c.get('oi', 0),
                    'c_iv': c.get('iv', 0),
                    'c_ltp': c.get('ltp', 0),
                    'p_ltp': p.get('ltp', 0),
                    'p_iv': p.get('iv', 0),
                    'p_oi': p.get('oi', 0),
                    'p_vol': p.get('vol', 0)
                })
            
            chain = pd.DataFrame(rows).set_index('strike')
            
            # 6. Filter ATM
            atm_strike = min(all_strikes, key=lambda x: abs(x - spot))
            atm_idx = all_strikes.index(atm_strike)
            start_idx = max(0, atm_idx - 8)
            end_idx = min(len(all_strikes), atm_idx + 9)
            filtered_strikes = all_strikes[start_idx:end_idx]
            
            chain = chain.loc[filtered_strikes]
            
            return chain, spot, exp_dt.strftime('%d-%b')
            
        except Exception as e:
            logger.error(f"Deribit Chain Error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0, ""


class BinanceDashboard:
    """Binance Options Chain - Public API"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def safe_float(self, val, default=0.0):
        if val is None:
            return default
        try:
            return float(val)
        except:
            return default

    def get_spot_price(self, symbol='BTCUSDT'):
        """Get spot price"""
        try:
            url = f"{BINANCE_URL}/eapi/v1/index"
            params = {'underlying': symbol.replace('USDT', '')}
            res = self.session.get(url, params=params, timeout=10)
            
            if res.status_code == 200:
                data = res.json()
                return self.safe_float(data.get('indexPrice'))
            return 0
        except Exception as e:
            logger.error(f"Binance Spot Error: {e}")
            return 0

    def get_chain_data(self, underlying='BTC'):
        """Fetch Binance option chain"""
        try:
            # 1. Get spot
            spot = self.get_spot_price(f"{underlying}USDT")
            if spot == 0:
                return None, 0, ""
            
            logger.info(f"💰 Binance {underlying} Spot: ${spot:,.2f}")

            # 2. Get exchange info for expiries
            info_url = f"{BINANCE_URL}/eapi/v1/exchangeInfo"
            info_res = self.session.get(info_url, timeout=10)
            
            if info_res.status_code != 200:
                return None, 0, ""
            
            info_data = info_res.json()
            symbols = info_data.get('optionSymbols', [])
            
            # Find nearest expiry
            expiries = {}
            now = datetime.utcnow()
            
            for sym in symbols:
                symbol = sym.get('symbol', '')
                if not symbol.startswith(underlying):
                    continue
                
                exp_date = sym.get('expiryDate')
                if exp_date:
                    try:
                        exp_ts = int(exp_date) / 1000
                        exp_dt = datetime.utcfromtimestamp(exp_ts)
                        if exp_dt > now:
                            date_str = exp_dt.strftime('%d%b%y').upper()
                            expiries[date_str] = exp_dt
                    except:
                        continue
            
            if not expiries:
                return None, 0, ""
            
            exp_str, exp_dt = sorted(expiries.items(), key=lambda x: x[1])[0]
            logger.info(f"📅 Binance Using expiry: {exp_str}")

            # 3. Get mark prices and OI
            mark_url = f"{BINANCE_URL}/eapi/v1/mark"
            mark_res = self.session.get(mark_url, timeout=15)
            
            if mark_res.status_code != 200:
                return None, 0, ""
            
            marks = mark_res.json()
            
            # 4. Get open interest
            oi_data = {}
            try:
                oi_url = f"{BINANCE_URL}/eapi/v1/openInterest"
                oi_params = {
                    'underlyingAsset': underlying,
                    'expirationDate': exp_str
                }
                oi_res = self.session.get(oi_url, params=oi_params, timeout=10)
                if oi_res.status_code == 200:
                    oi_list = oi_res.json()
                    for item in oi_list:
                        symbol = item.get('symbol', '')
                        oi_data[symbol] = self.safe_float(item.get('sumOpenInterest'))
            except:
                pass

            # 5. Parse data
            calls_data = {}
            puts_data = {}
            
            for mark in marks:
                symbol = mark.get('symbol', '')
                
                # Filter by underlying and expiry
                if not symbol.startswith(underlying):
                    continue
                
                # Parse: BTC-241227-90000-C
                parts = symbol.split('-')
                if len(parts) != 4:
                    continue
                
                exp_part = parts[1]
                strike = self.safe_float(parts[2])
                opt_type = parts[3]
                
                # Check expiry match
                try:
                    exp_check = datetime.strptime(exp_part, '%y%m%d').strftime('%d%b%y').upper()
                    if exp_check != exp_str:
                        continue
                except:
                    continue
                
                # Extract data
                data = {
                    'ltp': self.safe_float(mark.get('markPrice')),
                    'oi': oi_data.get(symbol, 0),
                    'vol': 0,  # Binance doesn't provide 24h volume easily
                    'iv': self.safe_float(mark.get('markIV')) * 100,  # Convert to %
                    'delta': self.safe_float(mark.get('delta')),
                    'gamma': self.safe_float(mark.get('gamma')),
                    'theta': self.safe_float(mark.get('theta')),
                    'vega': self.safe_float(mark.get('vega'))
                }
                
                if opt_type == 'C':
                    calls_data[strike] = data
                else:
                    puts_data[strike] = data
            
            logger.info(f"✅ Parsed: {len(calls_data)} calls, {len(puts_data)} puts")
            
            # 6. Build DataFrame
            all_strikes = sorted(set(list(calls_data.keys()) + list(puts_data.keys())))
            
            rows = []
            for strike in all_strikes:
                c = calls_data.get(strike, {})
                p = puts_data.get(strike, {})
                
                rows.append({
                    'strike': strike,
                    'c_vol': c.get('vol', 0),
                    'c_oi': c.get('oi', 0),
                    'c_iv': c.get('iv', 0),
                    'c_ltp': c.get('ltp', 0),
                    'p_ltp': p.get('ltp', 0),
                    'p_iv': p.get('iv', 0),
                    'p_oi': p.get('oi', 0),
                    'p_vol': p.get('vol', 0)
                })
            
            chain = pd.DataFrame(rows).set_index('strike')
            
            # 7. Filter ATM
            atm_strike = min(all_strikes, key=lambda x: abs(x - spot))
            atm_idx = all_strikes.index(atm_strike)
            start_idx = max(0, atm_idx - 8)
            end_idx = min(len(all_strikes), atm_idx + 9)
            filtered_strikes = all_strikes[start_idx:end_idx]
            
            chain = chain.loc[filtered_strikes]
            
            return chain, spot, exp_dt.strftime('%d-%b')
            
        except Exception as e:
            logger.error(f"Binance Chain Error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0, ""


def format_value(val, is_price=False, is_iv=False):
    """Format values"""
    if val == 0 or pd.isna(val):
        return "-"
    
    if is_iv:
        return f"{val:.1f}"
    
    if is_price:
        if val >= 1000:
            return f"{val:,.0f}"
        elif val >= 1:
            return f"{val:.2f}"
        else:
            return f"{val:.4f}"
    
    # OI/Volume
    if val >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    elif val >= 1_000:
        return f"{val/1_000:.1f}K"
    else:
        return f"{val:.0f}"


def generate_dashboard(exchange, chain_df, spot, exp, underlying='BTC'):
    """Generate option chain table image"""
    
    if chain_df is None or chain_df.empty:
        return None
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    title = f"{exchange.upper()} - {underlying} OPTION CHAIN (Exp: {exp}) | Spot: ${spot:,.2f}"
    ax.set_title(title, color='yellow', fontsize=16, pad=10)

    table_data = []
    col_labels = ['Vol', 'OI', 'IV%', 'LTP', 'STRIKE', 'LTP', 'IV%', 'OI', 'Vol']

    for strike in chain_df.index:
        row = chain_df.loc[strike]
        
        r = [
            format_value(row['c_vol']),
            format_value(row['c_oi']),
            format_value(row['c_iv'], is_iv=True),
            format_value(row['c_ltp'], is_price=True),
            f"{int(strike):,}",
            format_value(row['p_ltp'], is_price=True),
            format_value(row['p_iv'], is_iv=True),
            format_value(row['p_oi']),
            format_value(row['p_vol'])
        ]
        table_data.append(r)

    if not table_data:
        return None

    table = ax.table(cellText=table_data, colLabels=col_labels, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)

    # Style
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#333333')
        else:
            cell.set_edgecolor('#555555')
            cell.set_facecolor('black')
            cell.set_text_props(color='white')

            # Highlight ATM
            try:
                stk_str = table_data[row-1][4].replace(',', '')
                stk = float(stk_str)
                if abs(stk - spot) < (spot * 0.01):
                    cell.set_facecolor('#2A2A4A')
                    cell.set_text_props(color='#00FFFF', weight='bold')
            except:
                pass

            # Colors
            if col < 4:
                cell.set_text_props(color='#00FF00')
            elif col > 4:
                cell.set_text_props(color='#FF5555')
            elif col == 4:
                cell.set_text_props(color='yellow', weight='bold')

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
               facecolor='black', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ==================== MAIN ====================
async def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN missing!")
        return

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    deribit = DeribitDashboard()
    binance = BinanceDashboard()
    
    logger.info("🚀 Multi-Exchange Dashboard Started...")

    while True:
        try:
            # DERIBIT BTC
            logger.info("\n" + "="*50)
            logger.info("Fetching DERIBIT BTC...")
            chain, spot, exp = deribit.get_chain_data('BTC')
            if chain is not None:
                img = generate_dashboard('deribit', chain, spot, exp, 'BTC')
                if img:
                    await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img,
                        caption="📊 #DERIBIT #BTC Option Chain")
                    logger.info("✅ Deribit BTC Sent")
            
            await asyncio.sleep(15)

            # BINANCE BTC
            logger.info("\n" + "="*50)
            logger.info("Fetching BINANCE BTC...")
            chain, spot, exp = binance.get_chain_data('BTC')
            if chain is not None:
                img = generate_dashboard('binance', chain, spot, exp, 'BTC')
                if img:
                    await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img,
                        caption="📊 #BINANCE #BTC Option Chain")
                    logger.info("✅ Binance BTC Sent")
            
            await asyncio.sleep(15)

            # DERIBIT ETH
            logger.info("\n" + "="*50)
            logger.info("Fetching DERIBIT ETH...")
            chain, spot, exp = deribit.get_chain_data('ETH')
            if chain is not None:
                img = generate_dashboard('deribit', chain, spot, exp, 'ETH')
                if img:
                    await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img,
                        caption="📊 #DERIBIT #ETH Option Chain")
                    logger.info("✅ Deribit ETH Sent")
            
            await asyncio.sleep(15)

            # BINANCE ETH
            logger.info("\n" + "="*50)
            logger.info("Fetching BINANCE ETH...")
            chain, spot, exp = binance.get_chain_data('ETH')
            if chain is not None:
                img = generate_dashboard('binance', chain, spot, exp, 'ETH')
                if img:
                    await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img,
                        caption="📊 #BINANCE #ETH Option Chain")
                    logger.info("✅ Binance ETH Sent")

            logger.info("\n💤 Waiting 5 minutes...")
            await asyncio.sleep(300)

        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    # Test mode
    if os.getenv('TEST_MODE'):
        print("=" * 60)
        print("TESTING DERIBIT")
        print("=" * 60)
        deribit = DeribitDashboard()
        chain, spot, exp = deribit.get_chain_data('BTC')
        if chain is not None:
            print(f"Spot: ${spot:,.2f}, Expiry: {exp}")
            print("\nSample data:")
            print(chain.head())
            img = generate_dashboard('deribit', chain, spot, exp, 'BTC')
            if img:
                with open('deribit_btc_test.png', 'wb') as f:
                    f.write(img.read())
                print("✅ Saved: deribit_btc_test.png")
        
        print("\n" + "=" * 60)
        print("TESTING BINANCE")
        print("=" * 60)
        binance = BinanceDashboard()
        chain, spot, exp = binance.get_chain_data('BTC')
        if chain is not None:
            print(f"Spot: ${spot:,.2f}, Expiry: {exp}")
            print("\nSample data:")
            print(chain.head())
            img = generate_dashboard('binance', chain, spot, exp, 'BTC')
            if img:
                with open('binance_btc_test.png', 'wb') as f:
                    f.write(img.read())
                print("✅ Saved: binance_btc_test.png")
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("Stopped by user")
