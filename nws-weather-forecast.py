#!/usr/bin/env python3
"""
NWS Weather Forecast Module for Kalshi Trading

Fetches official NWS forecasts and calculates probabilities for Kalshi weather markets.

Key insight from PredictionArena research:
- NWS forecasts are accurate within ±2-3°F near settlement (<48h)
- High-probability outcomes are systematically underpriced (favorite-longshot bias)
- Example: 75% probability events priced at 40-50¢

Author: Clawd
Date: 2026-01-29
"""

import requests
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import math

# Cache file for forecasts (avoid hammering NWS API)
FORECAST_CACHE_FILE = "data/weather/nws-forecast-cache.json"
CACHE_TTL_MINUTES = 30  # Refresh every 30 minutes

# NWS API endpoints by city
NWS_POINTS = {
    "NYC": {"lat": 40.7829, "lon": -73.9654, "name": "Central Park"},  # Official NYC temp station
    "MIA": {"lat": 25.7959, "lon": -80.2870, "name": "Miami International Airport"},
    "DEN": {"lat": 39.8561, "lon": -104.6737, "name": "Denver International Airport"},
    "CHI": {"lat": 41.9692, "lon": -87.9073, "name": "O'Hare International Airport"},
    "LAX": {"lat": 33.9425, "lon": -118.4081, "name": "Los Angeles International Airport"},
    "HOU": {"lat": 29.9844, "lon": -95.3414, "name": "Houston Intercontinental Airport"},
    "AUS": {"lat": 30.1945, "lon": -97.6699, "name": "Austin-Bergstrom Airport"},
    "PHI": {"lat": 39.8729, "lon": -75.2437, "name": "Philadelphia International Airport"},
    "SFO": {"lat": 37.6213, "lon": -122.3790, "name": "San Francisco International Airport"},
}

# Forecast uncertainty (standard deviation in °F)
# UPDATED 2026-02-08: Actual MAE is 2.81°F. Using higher uncertainty for conservative edge calculation.
# Previous values (2.0-2.5°F same-day) led to overconfident probability estimates and 17.9% win rate.
# New values use 1.5-2x the observed MAE to properly capture distribution tails.
FORECAST_UNCERTAINTY = {
    "day_0": 4.0,   # Same day: ±4°F (was 2.0, actual MAE 2.81)
    "day_1": 5.0,   # Next day: ±5°F (was 2.5)
    "day_2": 6.0,   # 2 days out: ±6°F (was 3.0)
    "day_3": 8.0,   # 3+ days out: ±8°F (was 4.0)
}

def get_nws_headers():
    """NWS requires User-Agent header"""
    return {
        "User-Agent": "(Kalshi-Weather-Trader, research@example.com)",
        "Accept": "application/geo+json"
    }

def get_forecast_grid(city: str) -> Optional[str]:
    """Get NWS forecast grid URL for a city"""
    if city not in NWS_POINTS:
        print(f"❌ Unknown city: {city}")
        return None
    
    point = NWS_POINTS[city]
    url = f"https://api.weather.gov/points/{point['lat']},{point['lon']}"
    
    try:
        resp = requests.get(url, headers=get_nws_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("properties", {}).get("forecast")
    except Exception as e:
        print(f"❌ Error getting grid for {city}: {e}")
    
    return None

def fetch_forecast(city: str) -> Optional[Dict]:
    """Fetch NWS forecast for a city"""
    
    # Check cache first
    cache = load_cache()
    if city in cache:
        cached = cache[city]
        cached_time = datetime.fromisoformat(cached["fetched_at"])
        if datetime.now(timezone.utc) - cached_time < timedelta(minutes=CACHE_TTL_MINUTES):
            return cached["forecast"]
    
    # Fetch fresh
    forecast_url = get_forecast_grid(city)
    if not forecast_url:
        return None
    
    try:
        resp = requests.get(forecast_url, headers=get_nws_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            periods = data.get("properties", {}).get("periods", [])
            
            forecast = {
                "city": city,
                "location": NWS_POINTS[city]["name"],
                "periods": []
            }
            
            for p in periods:
                forecast["periods"].append({
                    "name": p.get("name"),
                    "start_time": p.get("startTime"),
                    "end_time": p.get("endTime"),
                    "temperature": p.get("temperature"),
                    "temperature_unit": p.get("temperatureUnit"),
                    "is_daytime": p.get("isDaytime"),
                    "short_forecast": p.get("shortForecast"),
                    "wind_speed": p.get("windSpeed"),
                    "wind_direction": p.get("windDirection"),
                })
            
            # Cache it
            save_to_cache(city, forecast)
            return forecast
    except Exception as e:
        print(f"❌ Error fetching forecast for {city}: {e}")
    
    return None

def load_cache() -> Dict:
    """Load forecast cache"""
    try:
        os.makedirs(os.path.dirname(FORECAST_CACHE_FILE), exist_ok=True)
        if os.path.exists(FORECAST_CACHE_FILE):
            with open(FORECAST_CACHE_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return {}

def save_to_cache(city: str, forecast: Dict):
    """Save forecast to cache"""
    try:
        cache = load_cache()
        cache[city] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "forecast": forecast
        }
        os.makedirs(os.path.dirname(FORECAST_CACHE_FILE), exist_ok=True)
        with open(FORECAST_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"⚠️ Cache save failed: {e}")

def get_forecast_for_date(city: str, target_date: datetime) -> Optional[Tuple[int, int, float]]:
    """
    Get high and low temperature forecast for a specific date.
    
    Returns: (high_temp, low_temp, uncertainty_std) or None
    """
    forecast = fetch_forecast(city)
    if not forecast:
        return None
    
    target_date_str = target_date.strftime("%Y-%m-%d")
    
    high_temp = None
    low_temp = None
    
    for period in forecast["periods"]:
        period_date = period["start_time"][:10]  # YYYY-MM-DD
        
        if period_date == target_date_str:
            temp = period["temperature"]
            if period["is_daytime"]:
                high_temp = temp
            else:
                low_temp = temp
    
    if high_temp is None and low_temp is None:
        return None
    
    # Calculate days until target
    now = datetime.now(timezone.utc)
    days_out = (target_date.date() - now.date()).days
    
    # Get uncertainty based on days out
    if days_out <= 0:
        uncertainty = FORECAST_UNCERTAINTY["day_0"]
    elif days_out == 1:
        uncertainty = FORECAST_UNCERTAINTY["day_1"]
    elif days_out == 2:
        uncertainty = FORECAST_UNCERTAINTY["day_2"]
    else:
        uncertainty = FORECAST_UNCERTAINTY["day_3"]
    
    return (high_temp, low_temp, uncertainty)

def calculate_probability(forecast_temp: float, uncertainty: float, 
                          lower_bound: float = None, upper_bound: float = None) -> float:
    """
    Calculate probability that actual temperature falls within bounds.
    
    Uses normal distribution centered on forecast with given uncertainty (std dev).
    
    Args:
        forecast_temp: Forecasted temperature
        uncertainty: Standard deviation (±uncertainty covers ~68% of outcomes)
        lower_bound: Lower bound of range (None = -infinity)
        upper_bound: Upper bound of range (None = +infinity)
    
    Returns: Probability (0.0 to 1.0)
    """
    from scipy import stats
    
    # Create normal distribution
    dist = stats.norm(loc=forecast_temp, scale=uncertainty)
    
    if lower_bound is None:
        lower_bound = float('-inf')
    if upper_bound is None:
        upper_bound = float('inf')
    
    # CDF gives P(X <= x), so P(lower < X <= upper) = CDF(upper) - CDF(lower)
    prob = dist.cdf(upper_bound) - dist.cdf(lower_bound)
    
    return prob

def calculate_probability_simple(forecast_temp: float, uncertainty: float,
                                  lower_bound: float = None, upper_bound: float = None) -> float:
    """
    Simple probability calculation without scipy.
    Uses approximation of normal distribution.
    """
    # Use error function approximation
    def erf(x):
        # Approximation of error function
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        return sign * y
    
    def norm_cdf(x, mu, sigma):
        return 0.5 * (1 + erf((x - mu) / (sigma * math.sqrt(2))))
    
    if lower_bound is None:
        lower_cdf = 0
    else:
        lower_cdf = norm_cdf(lower_bound, forecast_temp, uncertainty)
    
    if upper_bound is None:
        upper_cdf = 1
    else:
        upper_cdf = norm_cdf(upper_bound, forecast_temp, uncertainty)
    
    return upper_cdf - lower_cdf

def parse_kalshi_weather_ticker(ticker: str, market_title: str = None) -> Optional[Dict]:
    """
    Parse Kalshi weather market ticker.
    
    Examples:
        KXHIGHNY-26JAN30-B17.5  -> High temp NYC, Jan 30 2026, bucket 17-18°
        KXHIGHMIA-26JAN30-T74   -> High temp Miami, Jan 30 2026, threshold >74°
        KXLOWTNYC-26JAN30-T5    -> Low temp NYC, Jan 30 2026, threshold <5°
    
    Note: Threshold direction (> or <) is determined from market_title if provided,
    otherwise inferred from typical temperature ranges.
    """
    import re
    
    # Pattern: SERIES-YYMONDD-TYPE
    # Type: B{num} = bucket (range), T{num} = threshold (above/below)
    
    pattern = r"^(KXHIGH|KXLOW|HIGH|LOW)([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})-(B|T)([\d.]+)$"
    match = re.match(pattern, ticker)
    
    if not match:
        return None
    
    temp_type = match.group(1)  # KXHIGH, KXLOW, etc.
    city = match.group(2)       # NY, MIA, etc.
    year = int("20" + match.group(3))
    month = match.group(4)      # JAN, FEB, etc.
    day = int(match.group(5))
    market_type = match.group(6)  # B or T
    value = float(match.group(7))
    
    # Parse month
    months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    month_num = months.get(month.upper())
    
    if not month_num:
        return None
    
    # Normalize city codes
    city_map = {"NY": "NYC", "MIA": "MIA", "DEN": "DEN", "CHI": "CHI", 
                "LAX": "LAX", "HOU": "HOU", "AUS": "AUS", "PHI": "PHI", "SFO": "SFO",
                "TNYC": "NYC"}  # LOWTNYC uses TNYC
    
    normalized_city = city_map.get(city, city)
    
    # Determine if high or low temp
    is_high = "HIGH" in temp_type
    
    # Determine threshold direction from market title or infer
    threshold_above = None  # True = ">N", False = "<N"
    
    if market_type == "B":
        # Bucket: value represents midpoint of 2-degree range
        # e.g., B17.5 = 17-18°, B71.5 = 71-72°
        lower_bound = value - 0.5
        upper_bound = value + 0.5
    else:
        # Threshold: determine direction from title or infer
        if market_title:
            # Check title for explicit direction
            if f">{value}" in market_title or f"> {value}" in market_title or f">{int(value)}" in market_title:
                threshold_above = True
            elif f"<{value}" in market_title or f"< {value}" in market_title or f"<{int(value)}" in market_title:
                threshold_above = False
        
        if threshold_above is None:
            # Infer from typical temperature ranges:
            # For high temps: extreme highs (>70-80°F) are "above", extreme lows (<20-30°F) are "below"
            # For low temps: similar logic
            if is_high:
                # High temp thresholds: values >50 are typically "above", <40 are typically "below"
                threshold_above = value >= 40
            else:
                # Low temp thresholds: values >30 are typically "above", <20 are typically "below"
                threshold_above = value >= 20
        
        if threshold_above:
            lower_bound = value  # Above this
            upper_bound = None
        else:
            lower_bound = None
            upper_bound = value  # Below this
    
    return {
        "ticker": ticker,
        "city": normalized_city,
        "date": datetime(year, month_num, day, tzinfo=timezone.utc),
        "is_high_temp": is_high,
        "market_type": "bucket" if market_type == "B" else "threshold",
        "threshold_above": threshold_above,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "midpoint": value,
    }

def calculate_weather_edge(market: Dict, yes_price_cents: int) -> Optional[Dict]:
    """
    Calculate edge for a weather market based on NWS forecast.
    
    Args:
        market: Parsed market info from parse_kalshi_weather_ticker()
        yes_price_cents: Current YES price in cents
    
    Returns: Dict with probability, edge, recommendation
    """
    city = market["city"]
    target_date = market["date"]
    
    # Get forecast
    forecast_data = get_forecast_for_date(city, target_date)
    if not forecast_data:
        return None
    
    high_temp, low_temp, uncertainty = forecast_data
    
    # Use high or low temp based on market type
    forecast_temp = high_temp if market["is_high_temp"] else low_temp
    
    if forecast_temp is None:
        return None
    
    # Calculate probability
    prob = calculate_probability_simple(
        forecast_temp, 
        uncertainty,
        market["lower_bound"],
        market["upper_bound"]
    )
    
    # Market implied probability
    market_prob = yes_price_cents / 100.0
    
    # Edge calculation
    edge_yes = prob - market_prob  # Positive = YES is underpriced
    edge_no = (1 - prob) - (1 - market_prob)  # Positive = NO is underpriced
    
    recommendation = None
    edge = 0
    
    if edge_yes > 0.10:  # 10% min edge
        recommendation = "BUY_YES"
        edge = edge_yes
    elif edge_no > 0.10:
        recommendation = "BUY_NO"
        edge = edge_no
    
    return {
        "ticker": market["ticker"],
        "city": city,
        "date": target_date.strftime("%Y-%m-%d"),
        "forecast_temp": forecast_temp,
        "uncertainty": uncertainty,
        "lower_bound": market["lower_bound"],
        "upper_bound": market["upper_bound"],
        "calculated_probability": round(prob, 4),
        "market_probability": market_prob,
        "edge_yes": round(edge_yes, 4),
        "edge_no": round(edge_no, 4),
        "recommendation": recommendation,
        "edge": round(edge, 4),
    }

def get_all_forecasts() -> Dict:
    """Get forecasts for all tracked cities"""
    forecasts = {}
    for city in NWS_POINTS:
        forecast = fetch_forecast(city)
        if forecast:
            forecasts[city] = forecast
            print(f"✅ {city} ({NWS_POINTS[city]['name']}): {len(forecast['periods'])} periods")
    return forecasts

if __name__ == "__main__":
    print("🌡️ NWS Weather Forecast Module")
    print("=" * 50)
    
    # Test fetch all forecasts
    print("\nFetching forecasts...")
    forecasts = get_all_forecasts()
    
    # Show NYC forecast
    if "NYC" in forecasts:
        print("\n📍 NYC Forecast:")
        for p in forecasts["NYC"]["periods"][:6]:
            print(f"   {p['name']}: {p['temperature']}°F - {p['short_forecast']}")
    
    # Test ticker parsing
    print("\n📊 Testing ticker parsing...")
    test_tickers = [
        "KXHIGHNY-26JAN30-B17.5",
        "KXHIGHMIA-26JAN30-T74",
        "KXLOWTNYC-26JAN30-T5",
    ]
    
    for ticker in test_tickers:
        parsed = parse_kalshi_weather_ticker(ticker)
        if parsed:
            print(f"   {ticker}:")
            print(f"      City: {parsed['city']}, Date: {parsed['date'].strftime('%Y-%m-%d')}")
            print(f"      Type: {'High' if parsed['is_high_temp'] else 'Low'} temp, {parsed['market_type']}")
            print(f"      Bounds: {parsed['lower_bound']} to {parsed['upper_bound']}")
    
    # Test edge calculation
    print("\n💰 Testing edge calculation...")
    market = parse_kalshi_weather_ticker("KXHIGHNY-26JAN30-B17.5")
    if market:
        edge_result = calculate_weather_edge(market, 39)  # YES bid 39¢
        if edge_result:
            print(f"   {edge_result['ticker']}:")
            print(f"      Forecast: {edge_result['forecast_temp']}°F ± {edge_result['uncertainty']}°F")
            print(f"      Our probability: {edge_result['calculated_probability']*100:.1f}%")
            print(f"      Market probability: {edge_result['market_probability']*100:.1f}%")
            print(f"      Edge YES: {edge_result['edge_yes']*100:.1f}%")
            print(f"      Edge NO: {edge_result['edge_no']*100:.1f}%")
            print(f"      Recommendation: {edge_result['recommendation'] or 'NO_TRADE'}")


def get_weather_market_direction(ticker: str) -> Optional[str]:
    """
    Fetch market from Kalshi API to determine threshold direction.
    
    Returns: "above" (>N), "below" (<N), or None
    """
    try:
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            market = data.get("market", {})
            title = market.get("title", "")
            
            # Check for direction indicators
            if ">" in title:
                return "above"
            elif "<" in title:
                return "below"
    except:
        pass
    return None


def analyze_weather_markets(city: str = "NYC") -> List[Dict]:
    """
    Analyze all open weather markets for a city and find edges.
    
    Returns list of opportunities with calculated edges.
    """
    opportunities = []
    
    # Map city to series tickers
    city_series = {
        "NYC": ["KXHIGHNY", "KXLOWTNYC"],
        "MIA": ["KXHIGHMIA", "KXLOWMIA"],
        "DEN": ["KXHIGHDEN"],
        "CHI": ["KXHIGHCHI"],
    }
    
    series_list = city_series.get(city, [])
    if not series_list:
        print(f"❌ No series mapping for {city}")
        return []
    
    for series in series_list:
        # Fetch markets
        url = "https://api.elections.kalshi.com/trade-api/v2/markets"
        params = {"series_ticker": series, "limit": 20, "status": "open"}
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                continue
            
            markets = resp.json().get("markets", [])
            
            for m in markets:
                ticker = m.get("ticker")
                title = m.get("title", "")
                yes_bid = m.get("yes_bid")
                yes_ask = m.get("yes_ask")
                
                if not ticker or yes_bid is None:
                    continue
                
                # Parse ticker with title for accurate direction
                parsed = parse_kalshi_weather_ticker(ticker, title)
                if not parsed:
                    continue
                
                # Calculate edge
                edge_result = calculate_weather_edge(parsed, yes_bid)
                if edge_result and edge_result.get("recommendation"):
                    edge_result["title"] = title
                    edge_result["yes_bid"] = yes_bid
                    edge_result["yes_ask"] = yes_ask
                    opportunities.append(edge_result)
        except Exception as e:
            print(f"⚠️ Error fetching {series}: {e}")
    
    # Sort by edge (highest first)
    opportunities.sort(key=lambda x: x.get("edge", 0), reverse=True)
    
    return opportunities


if __name__ == "__main__":
    print("🌡️ NWS Weather Forecast Module")
    print("=" * 50)
    
    # Analyze NYC weather markets
    print("\n🔍 Analyzing NYC weather markets...")
    opps = analyze_weather_markets("NYC")
    
    if opps:
        print(f"\n📊 Found {len(opps)} opportunities with edge:")
        for opp in opps[:10]:
            rec = opp.get("recommendation", "")
            ticker = opp.get("ticker", "")
            edge = opp.get("edge", 0) * 100
            prob = opp.get("calculated_probability", 0) * 100
            mkt_prob = opp.get("market_probability", 0) * 100
            forecast = opp.get("forecast_temp", "?")
            
            print(f"\n   {rec}: {ticker}")
            print(f"      Forecast: {forecast}°F")
            print(f"      Our prob: {prob:.1f}% vs Market: {mkt_prob:.1f}%")
            print(f"      Edge: {edge:+.1f}%")
    else:
        print("   No opportunities found with sufficient edge")
    
    # Also check Miami
    print("\n🔍 Analyzing Miami weather markets...")
    opps_mia = analyze_weather_markets("MIA")
    if opps_mia:
        print(f"\n📊 Found {len(opps_mia)} opportunities:")
        for opp in opps_mia[:5]:
            print(f"   {opp.get('recommendation')}: {opp.get('ticker')} - Edge: {opp.get('edge', 0)*100:+.1f}%")
