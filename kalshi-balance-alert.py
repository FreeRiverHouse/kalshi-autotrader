#!/usr/bin/env python3
"""
Kalshi Balance Alert - Checks balance and creates alert file if low.
Used by heartbeat to send Telegram notifications.

Usage:
    python kalshi-balance-alert.py             # Check and update alert status
    python kalshi-balance-alert.py --status    # Just show status
"""

import requests
import json
import sys
import os
from datetime import datetime, timezone
import base64
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ============== CONFIG ==============
# Credentials — loaded from environment or .kalshi-private-key.pem
API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID", "4308d1ca-585e-4b73-be82-5c0968b9a59a")
_key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.kalshi-private-key.pem')
if os.path.exists(_key_file):
    with open(_key_file) as _f:
        PRIVATE_KEY = _f.read().strip()
elif os.environ.get("KALSHI_PRIVATE_KEY"):
    PRIVATE_KEY = os.environ["KALSHI_PRIVATE_KEY"]
else:
    print("❌ Kalshi private key not found!")
    sys.exit(1)

BASE_URL = "https://api.elections.kalshi.com"
LOW_BALANCE_THRESHOLD_CENTS = 500  # $5.00
ALERT_FILE = Path(__file__).parent / "kalshi-low-balance.alert"


def sign_request(method: str, path: str, timestamp: str) -> str:
    """Sign request with RSA-PSS"""
    private_key = serialization.load_pem_private_key(PRIVATE_KEY.encode(), password=None)
    message = f"{timestamp}{method}{path}".encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')


def api_request(method: str, path: str) -> dict:
    """Make authenticated API request"""
    timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
    signature = sign_request(method, path, timestamp)
    headers = {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}{path}"
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def get_balance() -> dict:
    """Get account balance"""
    return api_request("GET", "/trade-api/v2/portfolio/balance")


def check_and_alert():
    """Check balance and update alert file"""
    balance = get_balance()
    
    if "error" in balance:
        print(f"❌ Error getting balance: {balance['error']}")
        return None
    
    cash_cents = balance.get("balance", 0)
    portfolio_cents = balance.get("portfolio_value", 0)
    total_cents = cash_cents + portfolio_cents
    
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cash_cents": cash_cents,
        "portfolio_cents": portfolio_cents,
        "total_cents": total_cents,
        "threshold_cents": LOW_BALANCE_THRESHOLD_CENTS,
        "is_low": cash_cents < LOW_BALANCE_THRESHOLD_CENTS
    }
    
    print(f"💰 Cash Balance: ${cash_cents/100:.2f}")
    print(f"📊 Portfolio Value: ${portfolio_cents/100:.2f}")
    print(f"📈 Total: ${total_cents/100:.2f}")
    print(f"⚠️ Threshold: ${LOW_BALANCE_THRESHOLD_CENTS/100:.2f}")
    
    if result["is_low"]:
        print(f"\n🚨 LOW BALANCE ALERT! Cash (${cash_cents/100:.2f}) < ${LOW_BALANCE_THRESHOLD_CENTS/100:.2f}")
        # Write alert file
        with open(ALERT_FILE, "w") as f:
            json.dump(result, f, indent=2)
        print(f"📝 Alert written to {ALERT_FILE}")
    else:
        print(f"\n✅ Balance OK")
        # Remove alert file if exists
        if ALERT_FILE.exists():
            ALERT_FILE.unlink()
            print(f"🗑️ Cleared old alert")
    
    return result


def get_status():
    """Just show current status without writing alerts"""
    balance = get_balance()
    
    if "error" in balance:
        print(f"❌ Error: {balance['error']}")
        return
    
    cash = balance.get("balance", 0)
    portfolio = balance.get("portfolio_value", 0)
    
    print(f"💰 Kalshi Balance Status")
    print(f"   Cash: ${cash/100:.2f}")
    print(f"   Portfolio: ${portfolio/100:.2f}")
    print(f"   Total: ${(cash+portfolio)/100:.2f}")
    
    if ALERT_FILE.exists():
        print(f"\n⚠️ Active low balance alert!")
    else:
        print(f"\n✅ No active alerts")


if __name__ == "__main__":
    if "--status" in sys.argv:
        get_status()
    else:
        check_and_alert()
