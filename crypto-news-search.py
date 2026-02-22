#!/usr/bin/env python3
"""
Crypto News Search - Fundamental Analysis for Kalshi Autotrader

Implements the "Grok Fundamental" strategy: use web search to find
informational edge before making trades.

Key news types that move crypto:
- Fed/FOMC announcements (rate decisions, speeches)
- ETF news (approvals, inflows/outflows)
- Exchange news (hacks, insolvencies, regulations)
- Macro events (CPI, unemployment, GDP)
- Whale movements, liquidations
- Protocol updates (halvings, upgrades)

Usage:
    # As module
    from crypto_news_search import get_crypto_sentiment
    sentiment = get_crypto_sentiment()  # Returns: bullish, bearish, or neutral
    
    # As CLI
    python crypto-news-search.py [--asset btc|eth|both] [--json]

Author: Clawd (implementing T661 - Grok Fundamental strategy)
"""

import os
import sys
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import subprocess
import hashlib

# Cache to avoid repeated searches
CACHE_FILE = str(Path(__file__).parent / "crypto-news-cache.json")
CACHE_TTL_MINUTES = 15  # Re-search every 15 minutes

# Keywords that indicate sentiment
BULLISH_KEYWORDS = [
    # Positive price action
    "rally", "surge", "soar", "jump", "spike", "breakout", "all-time high", "ATH",
    "bull run", "bullish", "moon", "pump",
    # Institutional adoption
    "ETF approved", "ETF inflow", "institutional buying", "MicroStrategy",
    "corporate adoption", "Tesla", "major investment",
    # Regulatory positive
    "SEC approval", "regulatory clarity", "legal win", "court victory",
    # Technical positive
    "halving", "upgrade successful", "network growth", "adoption",
    # Macro positive
    "rate cut", "dovish Fed", "inflation cooling", "soft landing",
]

BEARISH_KEYWORDS = [
    # Negative price action
    "crash", "plunge", "dump", "selloff", "sell-off", "collapse", "tank",
    "bearish", "correction", "liquidation", "capitulation",
    # Exchange/custody issues
    "hack", "exploit", "breach", "insolvency", "bankruptcy", "FTX",
    "exchange down", "withdrawal halt", "frozen",
    # Regulatory negative
    "SEC lawsuit", "ban", "crackdown", "investigation", "enforcement",
    "China ban", "regulatory concerns",
    # Technical negative
    "51% attack", "vulnerability", "bug", "fork dispute",
    # Macro negative
    "rate hike", "hawkish Fed", "inflation hot", "recession fears",
    "bank crisis", "contagion",
]

# High-impact events (override normal sentiment)
HIGH_IMPACT_BULLISH = [
    "Bitcoin ETF approved",
    "Ethereum ETF approved",
    "Fed cuts rates",
    "FOMC dovish",
    "major corporate buy",
]

HIGH_IMPACT_BEARISH = [
    "major exchange hack",
    "exchange insolvency",
    "SEC sues",
    "Fed emergency",
    "rate hike surprise",
    "Tether concerns",
]


def load_cache() -> Dict:
    """Load cached search results."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def save_cache(cache: Dict):
    """Save search results to cache."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}", file=sys.stderr)


def is_cache_valid(cache: Dict, key: str) -> bool:
    """Check if cached result is still valid."""
    if key not in cache:
        return False
    
    cached_time = datetime.fromisoformat(cache[key].get("timestamp", "2000-01-01T00:00:00"))
    age_minutes = (datetime.now(timezone.utc).replace(tzinfo=None) - cached_time).total_seconds() / 60
    return age_minutes < CACHE_TTL_MINUTES


def search_crypto_news(query: str, count: int = 5) -> List[Dict]:
    """
    Search for crypto news using Clawdbot's web_search capability.
    
    Priority:
    1. Brave Search API (if key available)
    2. RSS feeds (free, no API key needed)
    """
    results = []
    
    # Try using a simple curl to Brave Search API if we have the key
    brave_key = os.getenv("BRAVE_API_KEY")
    
    if brave_key:
        try:
            import urllib.request
            import urllib.parse
            
            encoded_query = urllib.parse.quote(query)
            url = f"https://api.search.brave.com/res/v1/web/search?q={encoded_query}&count={count}&freshness=pd"
            
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            req.add_header("X-Subscription-Token", brave_key)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
            for result in data.get("web", {}).get("results", [])[:count]:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "description": result.get("description", ""),
                })
            if results:
                return results
        except Exception as e:
            print(f"Brave API error: {e}", file=sys.stderr)
    
    # Fallback: Use RSS feeds (T420)
    try:
        # Try to import and use the RSS fetcher
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rss_script = os.path.join(script_dir, "fetch-crypto-rss.py")
        
        if os.path.exists(rss_script):
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location("fetch_crypto_rss", rss_script)
            rss_module = module_from_spec(spec)
            spec.loader.exec_module(rss_module)
            
            # Get recent headlines
            data = rss_module.fetch_all_feeds()
            for article in data.get("results", [])[:count]:
                results.append({
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "description": article.get("description", ""),
                })
            if results:
                return results
    except Exception as e:
        print(f"RSS feed error: {e}", file=sys.stderr)
    
    # Final fallback: Check cached RSS data directly
    try:
        news_cache = "data/crypto-news-feed.json"
        if os.path.exists(news_cache):
            with open(news_cache, 'r') as f:
                cached = json.load(f)
            # Return cached headlines if recent (within 2 hours)
            if cached.get("timestamp"):
                cached_time = datetime.fromisoformat(cached["timestamp"].replace('Z', '+00:00'))
                age = (datetime.now(timezone.utc) - cached_time).total_seconds()
                if age < 7200:  # 2 hours
                    for article in cached.get("results", [])[:count]:
                        results.append({
                            "title": article.get("title", ""),
                            "url": article.get("url", ""),
                            "description": article.get("description", ""),
                        })
    except Exception as e:
        print(f"Cache read error: {e}", file=sys.stderr)
    
    return results


def analyze_sentiment(texts: List[str]) -> Tuple[str, float, List[str]]:
    """
    Analyze sentiment from a list of news headlines/descriptions.
    
    Returns: (sentiment, confidence, reasons)
    - sentiment: 'bullish', 'bearish', or 'neutral'
    - confidence: 0.0 to 1.0
    - reasons: list of matched keywords/phrases
    """
    bullish_matches = []
    bearish_matches = []
    
    full_text = " ".join(texts).lower()
    
    # Check for high-impact events first
    for phrase in HIGH_IMPACT_BULLISH:
        if phrase.lower() in full_text:
            return ("bullish", 0.95, [f"HIGH IMPACT: {phrase}"])
    
    for phrase in HIGH_IMPACT_BEARISH:
        if phrase.lower() in full_text:
            return ("bearish", 0.95, [f"HIGH IMPACT: {phrase}"])
    
    # Count keyword matches
    for keyword in BULLISH_KEYWORDS:
        if keyword.lower() in full_text:
            bullish_matches.append(keyword)
    
    for keyword in BEARISH_KEYWORDS:
        if keyword.lower() in full_text:
            bearish_matches.append(keyword)
    
    # Calculate sentiment
    total_matches = len(bullish_matches) + len(bearish_matches)
    
    if total_matches == 0:
        return ("neutral", 0.5, ["No significant news detected"])
    
    bullish_ratio = len(bullish_matches) / total_matches
    
    if bullish_ratio > 0.65:
        confidence = min(0.9, 0.5 + (bullish_ratio - 0.5) * 0.8)
        return ("bullish", confidence, bullish_matches[:5])
    elif bullish_ratio < 0.35:
        confidence = min(0.9, 0.5 + (0.5 - bullish_ratio) * 0.8)
        return ("bearish", confidence, bearish_matches[:5])
    else:
        return ("neutral", 0.5, bullish_matches[:2] + bearish_matches[:2])


def get_scheduled_events() -> List[Dict]:
    """
    Check for known scheduled events that could impact crypto.
    
    Sources:
    - FOMC meetings (8 per year)
    - CPI releases (monthly)
    - Jobs reports (monthly)
    - Bitcoin halving (every ~4 years)
    """
    events = []
    now = datetime.now(timezone.utc)
    
    # Known 2024-2025 FOMC dates (approximate)
    fomc_dates = [
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    ]
    
    for date_str in fomc_dates:
        event_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        days_until = (event_date - now).days
        
        if 0 <= days_until <= 1:
            events.append({
                "type": "FOMC",
                "date": date_str,
                "days_until": days_until,
                "impact": "high",
                "note": "Fed rate decision - expect volatility"
            })
    
    # CPI typically released mid-month
    # Check if we're within 1 day of the 13th (approximate CPI date)
    if 12 <= now.day <= 14:
        events.append({
            "type": "CPI",
            "date": now.strftime("%Y-%m-%d"),
            "days_until": 0,
            "impact": "high",
            "note": "CPI release day - inflation data moves markets"
        })
    
    return events


def get_crypto_sentiment(asset: str = "both") -> Dict:
    """
    Main function: Get current crypto sentiment based on news search.
    
    Args:
        asset: 'btc', 'eth', or 'both'
    
    Returns:
        {
            "sentiment": "bullish" | "bearish" | "neutral",
            "confidence": 0.0-1.0,
            "reasons": [...],
            "events": [...],
            "timestamp": "...",
            "should_trade": True | False,
            "edge_adjustment": -0.05 to +0.05
        }
    """
    cache = load_cache()
    cache_key = f"sentiment_{asset}"
    
    # Check cache
    if is_cache_valid(cache, cache_key):
        return cache[cache_key]["result"]
    
    # Build search queries based on asset
    queries = []
    if asset in ("btc", "both"):
        queries.append("Bitcoin BTC price news today")
    if asset in ("eth", "both"):
        queries.append("Ethereum ETH price news today")
    if asset == "both":
        queries.append("crypto market news Fed rates")
    
    # Collect all news
    all_texts = []
    for query in queries:
        results = search_crypto_news(query, count=3)
        for r in results:
            all_texts.append(r.get("title", "") + " " + r.get("description", ""))
    
    # Analyze sentiment
    sentiment, confidence, reasons = analyze_sentiment(all_texts)
    
    # Check scheduled events
    events = get_scheduled_events()
    event_warning = None
    
    for event in events:
        if event["impact"] == "high" and event["days_until"] <= 1:
            event_warning = f"⚠️ {event['type']} {event['note']}"
            # High-impact events = reduce confidence, suggest caution
            confidence *= 0.7
    
    # Calculate edge adjustment
    # Positive = boost YES bets, Negative = boost NO bets
    edge_adjustment = 0.0
    if sentiment == "bullish" and confidence > 0.6:
        edge_adjustment = 0.02 * confidence  # Up to +2% edge for YES
    elif sentiment == "bearish" and confidence > 0.6:
        edge_adjustment = -0.02 * confidence  # Up to +2% edge for NO
    
    # Should we trade?
    # Avoid trading during high-uncertainty events unless sentiment is clear
    should_trade = True
    if events and confidence < 0.7:
        should_trade = False
        reasons.append("⚠️ High-impact event approaching - recommend caution")
    
    result = {
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "reasons": reasons,
        "events": events,
        "event_warning": event_warning,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "should_trade": should_trade,
        "edge_adjustment": round(edge_adjustment, 4),
        "news_count": len(all_texts),
    }
    
    # Cache result
    cache[cache_key] = {
        "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        "result": result
    }
    save_cache(cache)
    
    return result


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto news sentiment analysis")
    parser.add_argument("--asset", choices=["btc", "eth", "both"], default="both",
                       help="Asset to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache")
    
    args = parser.parse_args()
    
    if args.no_cache and os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    
    result = get_crypto_sentiment(args.asset)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n📰 Crypto News Sentiment Analysis")
        print(f"{'='*40}")
        print(f"Asset: {args.asset.upper()}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Edge adjustment: {result['edge_adjustment']*100:+.2f}%")
        print(f"Should trade: {'✅ Yes' if result['should_trade'] else '⚠️ Caution'}")
        print(f"\nReasons:")
        for r in result['reasons']:
            print(f"  • {r}")
        if result['events']:
            print(f"\n⚠️ Scheduled Events:")
            for e in result['events']:
                print(f"  • {e['type']}: {e['note']}")
        print(f"\nTimestamp: {result['timestamp']}")
        print(f"News articles analyzed: {result['news_count']}")


if __name__ == "__main__":
    main()
