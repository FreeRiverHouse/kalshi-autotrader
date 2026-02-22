#!/usr/bin/env python3
"""
Kalshi Settlement Tracker
Tracks BTC/ETH/SOL price at contract expiry times to calculate actual win/loss.

Contract ticker format: KX[ASSET]D-[DATE][HOUR]-T[STRIKE]
BTC Example: KXBTCD-26JAN2804-T88499.99 = Jan 28 2026, 4:00 UTC, strike $88,500
ETH Example: KXETHD-26JAN2804-T3199.99 = Jan 28 2026, 4:00 UTC, strike $3,200
SOL Example: KXSOLD-26JAN2804-T249.99 = Jan 28 2026, 4:00 UTC, strike $250 (T423)
"""

import json
import sys
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import urllib.request
import urllib.error

TRADES_FILE = Path(__file__).parent / "kalshi-trades.jsonl"
TRADES_FILE_V2 = Path(__file__).parent / "kalshi-trades-v2.jsonl"
SETTLEMENTS_FILE = Path(__file__).parent / "kalshi-settlements.json"
SETTLEMENTS_FILE_V2 = Path(__file__).parent / "kalshi-settlements-v2.json"

def parse_ticker(ticker: str) -> dict:
    """
    Parse Kalshi BTC/ETH/SOL ticker to extract expiry time, strike, and asset.
    BTC Example: KXBTCD-26JAN2804-T88499.99
    ETH Example: KXETHD-26JAN2804-T3199.99
    SOL Example: KXSOLD-26JAN2804-T249.99 (T423)
    
    Format: KX[ASSET]D-[YY][MMM][DD][HH]-T[STRIKE]
    where YY=year (20YY), MMM=month, DD=day, HH=hour (ET timezone)
    """
    # Try BTC ticker format
    match = re.match(r'KXBTCD-(\d{2})([A-Z]{3})(\d{2})(\d{2})-T(\d+\.?\d*)', ticker)
    asset = 'BTC'
    
    # Try ETH ticker format if BTC didn't match
    if not match:
        match = re.match(r'KXETHD-(\d{2})([A-Z]{3})(\d{2})(\d{2})-T(\d+\.?\d*)', ticker)
        asset = 'ETH'
    
    # Try SOL ticker format if neither BTC nor ETH matched (T423)
    if not match:
        match = re.match(r'KXSOLD-(\d{2})([A-Z]{3})(\d{2})(\d{2})-T(\d+\.?\d*)', ticker)
        asset = 'SOL'
    
    if not match:
        return None
    
    year_short, month_str, day, hour_et, strike = match.groups()
    
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    
    year = 2000 + int(year_short)
    month = month_map.get(month_str, 1)
    day = int(day)
    hour_et = int(hour_et)
    
    # Kalshi uses ET (Eastern Time) for contract expiry
    # January = EST (UTC-5), so add 5 hours to get UTC
    # March-Nov (mostly) = EDT (UTC-4)
    # For simplicity, assume EST (UTC-5) for now
    hour_utc = hour_et + 5
    day_adj = day
    month_adj = month
    year_adj = year
    
    if hour_utc >= 24:
        hour_utc -= 24
        day_adj += 1
        
        # Handle month rollover
        import calendar
        days_in_month = calendar.monthrange(year_adj, month_adj)[1]
        if day_adj > days_in_month:
            day_adj = 1
            month_adj += 1
            if month_adj > 12:
                month_adj = 1
                year_adj += 1
    
    expiry_time = datetime(year_adj, month_adj, day_adj, hour_utc, 0, 0, tzinfo=timezone.utc)
    
    return {
        'expiry_time': expiry_time,
        'strike': float(strike),
        'expiry_hour_et': hour_et,
        'expiry_date': f"{year}-{month:02d}-{day:02d}",
        'asset': asset
    }


def get_price_at_time(target_time: datetime, asset: str = 'BTC') -> float:
    """
    Get crypto price at a specific historical time using CoinGecko API.
    Uses hourly granularity for accuracy.
    Supports BTC, ETH, and SOL (T423).
    """
    # CoinGecko market_chart/range endpoint
    # from/to are UNIX timestamps
    start_ts = int((target_time - timedelta(minutes=5)).timestamp())
    end_ts = int((target_time + timedelta(minutes=5)).timestamp())
    
    # Map asset to CoinGecko coin_id (T423: added SOL)
    coin_map = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana'}
    coin_id = coin_map.get(asset, 'bitcoin')
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency=usd&from={start_ts}&to={end_ts}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            
            if not data.get('prices'):
                return None
            
            # Find price closest to target time
            target_ts = target_time.timestamp() * 1000  # CoinGecko uses milliseconds
            prices = data['prices']
            
            closest = min(prices, key=lambda p: abs(p[0] - target_ts))
            return closest[1]
            
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  Error fetching price: {e}")
        return None


def get_price_binance(target_time: datetime, asset: str = 'BTC') -> float:
    """
    Get crypto price at a specific time using Binance klines API (backup).
    Supports BTC, ETH, and SOL (T423).
    """
    start_ts = int(target_time.timestamp() * 1000)
    end_ts = start_ts + 60000  # 1 minute later
    
    # Map asset to Binance symbol (T423: added SOL)
    symbol_map = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT'}
    symbol = symbol_map.get(asset, 'BTCUSDT')
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&startTime={start_ts}&endTime={end_ts}&limit=1"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            
            if data and len(data) > 0:
                # Kline format: [open_time, open, high, low, close, ...]
                # Use close price
                return float(data[0][4])
            return None
            
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  Binance error: {e}")
        return None


def get_price_cryptocompare(target_time: datetime, asset: str = 'BTC') -> float:
    """
    Get crypto price at a specific time using CryptoCompare API (no auth needed).
    Supports BTC, ETH, and SOL (T423 - uses fsym directly).
    """
    ts = int(target_time.timestamp())
    
    url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={asset}&tsym=USD&limit=1&toTs={ts}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            
            if data.get('Response') == 'Success' and data.get('Data', {}).get('Data'):
                # Use close price from the last data point
                prices = data['Data']['Data']
                if prices:
                    return float(prices[-1]['close'])
            return None
            
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  CryptoCompare error: {e}")
        return None


def determine_outcome(trade: dict, settlement_price: float) -> dict:
    """
    Determine if a trade won or lost based on settlement price.
    
    For BTC contracts:
    - YES wins if final_price >= strike
    - NO wins if final_price < strike
    """
    strike = trade.get('strike')
    side = trade.get('side', '').lower()
    contracts = trade.get('contracts', 0)
    price_cents = trade.get('price_cents', 0)
    cost_cents = trade.get('cost_cents', 0)
    
    if settlement_price >= strike:
        # YES wins
        if side == 'yes':
            won = True
            payout_cents = contracts * 100  # $1 per contract
            pnl_cents = payout_cents - cost_cents
        else:  # NO loses
            won = False
            payout_cents = 0
            pnl_cents = -cost_cents
    else:
        # NO wins
        if side == 'no':
            won = True
            payout_cents = contracts * 100
            pnl_cents = payout_cents - cost_cents
        else:  # YES loses
            won = False
            payout_cents = 0
            pnl_cents = -cost_cents
    
    return {
        'won': won,
        'payout_cents': payout_cents,
        'pnl_cents': pnl_cents,
        'settlement_price': settlement_price,
        'strike': strike,
        'side': side
    }


def load_settlements(settlements_file: Path = None) -> dict:
    """Load existing settlements data."""
    if settlements_file is None:
        settlements_file = SETTLEMENTS_FILE
    if settlements_file.exists():
        try:
            with open(settlements_file) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_settlements(data: dict, settlements_file: Path = None):
    """Save settlements data."""
    if settlements_file is None:
        settlements_file = SETTLEMENTS_FILE
    with open(settlements_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def update_trade_log_results(settlements: dict, trades_file: Path = None):
    """
    Update trade log file with settlement results.
    Rewrites the file, updating result_status for settled trades.
    """
    if trades_file is None:
        trades_file = TRADES_FILE
    if not trades_file.exists():
        return 0
    
    # Build lookup of settled tickers
    settled_results = {}
    for ticker, data in settlements.get('trades', {}).items():
        if data.get('status') == 'settled':
            settled_results[ticker] = 'won' if data.get('won') else 'lost'
    
    if not settled_results:
        return 0
    
    # Read all lines
    lines = []
    updated_count = 0
    with open(trades_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Check if this is a trade that needs updating
                if (entry.get('type') == 'trade' and 
                    entry.get('ticker') in settled_results and
                    entry.get('result_status') == 'pending'):
                    entry['result_status'] = settled_results[entry['ticker']]
                    updated_count += 1
                lines.append(json.dumps(entry))
            except json.JSONDecodeError:
                lines.append(line)
    
    # Write back
    if updated_count > 0:
        with open(trades_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        print(f"\nUpdated {updated_count} trade log entries ({trades_file.name}) with settlement results.")
    
    return updated_count


def process_settlements(trades_file: Path = None, settlements_file: Path = None):
    """Process all pending trades and determine outcomes."""
    if trades_file is None:
        trades_file = TRADES_FILE
    if settlements_file is None:
        settlements_file = SETTLEMENTS_FILE
    
    if not trades_file.exists():
        print(f"Trades file not found: {trades_file}")
        return {}
    
    source_name = trades_file.name
    print(f"\n{'='*50}")
    print(f"Processing: {source_name}")
    print(f"{'='*50}")
    
    now = datetime.now(timezone.utc)
    settlements = load_settlements(settlements_file)
    
    if 'trades' not in settlements:
        settlements['trades'] = {}
    if 'summary' not in settlements:
        settlements['summary'] = {'wins': 0, 'losses': 0, 'pending': 0, 'total_pnl_cents': 0}
    
    # Load all trades
    pending_trades = []
    with open(trades_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get('type') == 'trade' and entry.get('order_status') == 'executed':
                    ticker = entry.get('ticker')
                    # Skip if already settled successfully
                    existing = settlements['trades'].get(ticker, {})
                    if existing.get('status') == 'settled':
                        continue
                    # Include if not processed or if price fetch failed (retry)
                    pending_trades.append(entry)
            except json.JSONDecodeError:
                continue
    
    if not pending_trades:
        print("No new pending trades to process.")
        print(f"\nCurrent stats:")
        print(f"  Wins: {settlements['summary']['wins']}")
        print(f"  Losses: {settlements['summary']['losses']}")
        print(f"  Pending: {settlements['summary']['pending']}")
        print(f"  Total PnL: ${settlements['summary']['total_pnl_cents']/100:.2f}")
        return settlements
    
    print(f"Processing {len(pending_trades)} pending trades...")
    
    newly_settled = 0
    still_pending = 0
    
    for trade in pending_trades:
        ticker = trade.get('ticker')
        parsed = parse_ticker(ticker)
        
        if not parsed:
            print(f"  Skipping unparseable ticker: {ticker}")
            continue
        
        expiry_time = parsed['expiry_time']
        
        # Check if expiry has passed
        if expiry_time > now:
            still_pending += 1
            settlements['trades'][ticker] = {
                'status': 'pending',
                'asset': parsed.get('asset', 'BTC'),
                'expiry_time': expiry_time.isoformat(),
                'trade': trade
            }
            continue
        
        # Expiry has passed - get settlement price
        asset = parsed.get('asset', 'BTC')
        print(f"\n  Processing: {ticker}")
        print(f"    Asset: {asset}")
        print(f"    Expiry: {expiry_time.isoformat()}")
        print(f"    Strike: ${parsed['strike']:,.2f}")
        print(f"    Side: {trade.get('side')} @ {trade.get('price_cents')}¢")
        
        # Try multiple sources to get settlement price
        # Add delay to avoid rate limiting
        time.sleep(2)
        settlement_price = get_price_at_time(expiry_time, asset)
        if settlement_price is None:
            time.sleep(1.5)
            settlement_price = get_price_cryptocompare(expiry_time, asset)
        if settlement_price is None:
            time.sleep(1)
            settlement_price = get_price_binance(expiry_time, asset)
        
        if settlement_price is None:
            print(f"    Could not get {asset} settlement price - marking for retry")
            settlements['trades'][ticker] = {
                'status': 'price_fetch_failed',
                'asset': asset,
                'expiry_time': expiry_time.isoformat(),
                'trade': trade
            }
            continue
        
        # Determine outcome
        outcome = determine_outcome(trade, settlement_price)
        
        print(f"    Settlement price: ${settlement_price:,.2f}")
        print(f"    Result: {'✅ WIN' if outcome['won'] else '❌ LOSS'}")
        print(f"    PnL: ${outcome['pnl_cents']/100:.2f}")
        
        settlements['trades'][ticker] = {
            'status': 'settled',
            'asset': asset,
            'expiry_time': expiry_time.isoformat(),
            'settlement_price': settlement_price,
            'won': outcome['won'],
            'pnl_cents': outcome['pnl_cents'],
            'payout_cents': outcome['payout_cents'],
            'trade': trade
        }
        
        # Update summary
        if outcome['won']:
            settlements['summary']['wins'] += 1
        else:
            settlements['summary']['losses'] += 1
        settlements['summary']['total_pnl_cents'] += outcome['pnl_cents']
        
        newly_settled += 1
    
    settlements['summary']['pending'] = still_pending
    settlements['summary']['last_updated'] = now.isoformat()
    
    # Save
    save_settlements(settlements, settlements_file)
    
    # Update trade log with results
    update_trade_log_results(settlements, trades_file)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Settlement Summary ({source_name})")
    print(f"{'='*50}")
    print(f"Newly settled: {newly_settled}")
    print(f"Still pending: {still_pending}")
    print(f"\nOverall Stats:")
    print(f"  Wins: {settlements['summary']['wins']}")
    print(f"  Losses: {settlements['summary']['losses']}")
    total = settlements['summary']['wins'] + settlements['summary']['losses']
    if total > 0:
        win_rate = settlements['summary']['wins'] / total * 100
        print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total PnL: ${settlements['summary']['total_pnl_cents']/100:.2f}")
    
    return settlements


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--stats':
        # Show stats for both v1 and v2
        for name, settlements_file in [("v1", SETTLEMENTS_FILE), ("v2", SETTLEMENTS_FILE_V2)]:
            if not settlements_file.exists():
                continue
            print(f"\n--- {name} Stats ---")
            settlements = load_settlements(settlements_file)
            summary = settlements.get('summary', {})
            print(f"Wins: {summary.get('wins', 0)}")
            print(f"Losses: {summary.get('losses', 0)}")
            total = summary.get('wins', 0) + summary.get('losses', 0)
            if total > 0:
                print(f"Win Rate: {summary.get('wins', 0) / total * 100:.1f}%")
            print(f"PnL: ${summary.get('total_pnl_cents', 0)/100:.2f}")
            print(f"Pending: {summary.get('pending', 0)}")
    elif len(sys.argv) > 1 and sys.argv[1] == '--v2':
        # Process only v2 file
        process_settlements(TRADES_FILE_V2, SETTLEMENTS_FILE_V2)
    elif len(sys.argv) > 1 and sys.argv[1] == '--v1':
        # Process only v1 file
        process_settlements(TRADES_FILE, SETTLEMENTS_FILE)
    else:
        # Process both v1 and v2 files
        process_settlements(TRADES_FILE, SETTLEMENTS_FILE)
        if TRADES_FILE_V2.exists():
            process_settlements(TRADES_FILE_V2, SETTLEMENTS_FILE_V2)


if __name__ == '__main__':
    main()
