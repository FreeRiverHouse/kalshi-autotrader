#!/usr/bin/env python3
"""
Check if today is a US market holiday.
Used to optionally pause autotrader during unusual market conditions.

Usage:
    python3 check-market-holiday.py          # Exit 0 if holiday, 1 if not
    python3 check-market-holiday.py --json   # Output JSON with holiday info
    python3 check-market-holiday.py --list   # List upcoming holidays

US Market Holidays (when crypto may have unusual patterns):
- New Year's Day
- Martin Luther King Jr. Day  
- Presidents' Day
- Memorial Day
- Independence Day (July 4th)
- Labor Day
- Thanksgiving Day
- Christmas Day
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta

# US Federal/Market holidays for 2024-2027
# Format: (month, day) for fixed holidays, or calculated dates
FIXED_HOLIDAYS = {
    "new_years": (1, 1),
    "independence_day": (7, 4),
    "veterans_day": (11, 11),  # Sometimes observed, not major for crypto
    "christmas": (12, 25),
}

def get_nth_weekday(year: int, month: int, nth: int, weekday: int) -> date:
    """Get nth occurrence of weekday in month (weekday: 0=Mon, 6=Sun)"""
    first_day = date(year, month, 1)
    first_weekday = first_day.weekday()
    
    # Days until first occurrence of target weekday
    days_until = (weekday - first_weekday) % 7
    first_occurrence = first_day + timedelta(days=days_until)
    
    # Add weeks for nth occurrence
    return first_occurrence + timedelta(weeks=nth-1)

def get_last_weekday(year: int, month: int, weekday: int) -> date:
    """Get last occurrence of weekday in month"""
    # Start from last day of month
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    
    # Find last occurrence
    days_back = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=days_back)

def get_us_holidays(year: int) -> list[tuple[date, str]]:
    """Get all US market holidays for a given year"""
    holidays = []
    
    # Fixed date holidays
    holidays.append((date(year, 1, 1), "New Year's Day"))
    holidays.append((date(year, 7, 4), "Independence Day"))
    holidays.append((date(year, 12, 25), "Christmas Day"))
    
    # Variable date holidays
    # MLK Day: 3rd Monday of January
    holidays.append((get_nth_weekday(year, 1, 3, 0), "Martin Luther King Jr. Day"))
    
    # Presidents' Day: 3rd Monday of February
    holidays.append((get_nth_weekday(year, 2, 3, 0), "Presidents' Day"))
    
    # Memorial Day: Last Monday of May
    holidays.append((get_last_weekday(year, 5, 0), "Memorial Day"))
    
    # Labor Day: 1st Monday of September
    holidays.append((get_nth_weekday(year, 9, 1, 0), "Labor Day"))
    
    # Thanksgiving: 4th Thursday of November
    holidays.append((get_nth_weekday(year, 11, 4, 3), "Thanksgiving Day"))
    
    # Day after Thanksgiving (often low volume)
    thanksgiving = get_nth_weekday(year, 11, 4, 3)
    holidays.append((thanksgiving + timedelta(days=1), "Day After Thanksgiving"))
    
    # Christmas Eve and New Year's Eve (often reduced trading)
    holidays.append((date(year, 12, 24), "Christmas Eve"))
    holidays.append((date(year, 12, 31), "New Year's Eve"))
    
    return sorted(holidays, key=lambda x: x[0])

def is_holiday(check_date: date = None) -> tuple[bool, str | None]:
    """Check if a date is a US market holiday"""
    if check_date is None:
        check_date = date.today()
    
    holidays = get_us_holidays(check_date.year)
    
    for holiday_date, holiday_name in holidays:
        if holiday_date == check_date:
            return True, holiday_name
    
    return False, None

def get_upcoming_holidays(days: int = 60) -> list[dict]:
    """Get upcoming holidays within specified days"""
    today = date.today()
    end_date = today + timedelta(days=days)
    
    upcoming = []
    
    # Check current year and possibly next year
    for year in [today.year, today.year + 1]:
        holidays = get_us_holidays(year)
        for holiday_date, holiday_name in holidays:
            if today <= holiday_date <= end_date:
                days_until = (holiday_date - today).days
                upcoming.append({
                    "date": holiday_date.isoformat(),
                    "name": holiday_name,
                    "days_until": days_until,
                    "day_of_week": holiday_date.strftime("%A")
                })
    
    return sorted(upcoming, key=lambda x: x["date"])

def main():
    parser = argparse.ArgumentParser(description="Check US market holidays")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--list", action="store_true", help="List upcoming holidays")
    parser.add_argument("--days", type=int, default=60, help="Days to look ahead for --list")
    parser.add_argument("--date", type=str, help="Check specific date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    if args.list:
        upcoming = get_upcoming_holidays(args.days)
        if args.json:
            print(json.dumps(upcoming, indent=2))
        else:
            print(f"📅 Upcoming US Market Holidays (next {args.days} days):")
            print()
            if not upcoming:
                print("  No holidays in the specified period")
            for h in upcoming:
                emoji = "🔴" if h["days_until"] == 0 else "🟡" if h["days_until"] <= 7 else "⚪"
                print(f"  {emoji} {h['date']} ({h['day_of_week']}): {h['name']}")
                if h["days_until"] == 0:
                    print(f"     ⚠️  TODAY!")
                elif h["days_until"] == 1:
                    print(f"     📌 Tomorrow")
                else:
                    print(f"     📆 In {h['days_until']} days")
        return 0
    
    # Check specific date or today
    check_date = date.today()
    if args.date:
        try:
            check_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Error: Invalid date format. Use YYYY-MM-DD", file=sys.stderr)
            return 2
    
    is_hol, holiday_name = is_holiday(check_date)
    
    if args.json:
        result = {
            "date": check_date.isoformat(),
            "is_holiday": is_hol,
            "holiday_name": holiday_name,
            "recommendation": "pause_trading" if is_hol else "normal_trading"
        }
        print(json.dumps(result, indent=2))
    else:
        if is_hol:
            print(f"🔴 {check_date.isoformat()} is a US market holiday: {holiday_name}")
            print("   ⚠️  Crypto markets may have unusual patterns (lower liquidity)")
            print("   💡 Recommendation: Consider pausing autotrader or reducing position sizes")
        else:
            print(f"🟢 {check_date.isoformat()} is not a US market holiday")
    
    # Exit code: 0 = is holiday, 1 = not holiday
    return 0 if is_hol else 1

if __name__ == "__main__":
    sys.exit(main())
