import requests
import json
import csv
from datetime import datetime, timezone, timedelta
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
CSV_FILENAME = "btc_one_day_price_history.csv"
START_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 4, 30, tzinfo=timezone.utc)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# Use a session for connection pooling
session = requests.Session()

def get_month_name(month_int):
    return datetime(2000, month_int, 1).strftime('%B').lower()

def get_event_data(slug):
    url = f"{GAMMA_API_BASE}/events/slug/{slug}"
    try:
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return None

def fetch_token_history(token_id, outcome_label, m_slug, start_ts, end_ts):
    url = f"{CLOB_API_BASE}/prices-history"
    params = {
        "market": token_id,
        "startTs": int(start_ts),
        "endTs": int(end_ts),
        "fidelity": 1 # 1 minute fidelity
    }
    
    try:
        response = session.get(url, params=params, timeout=15)
        if response.status_code == 200:
            history = response.json().get("history", [])
            results = []
            for entry in history:
                results.append({
                    'market_slug': m_slug,
                    'outcome': outcome_label,
                    'timestamp': entry['t'],
                    'price': entry['p'],
                    'datetime_utc': datetime.fromtimestamp(entry['t'], tz=timezone.utc).isoformat()
                })
            return results
    except Exception as e:
        print(f"Error fetching {m_slug} ({outcome_label}): {e}")
    return []

def fetch_market_history(market_info, start_ts, end_ts):
    m_slug = market_info.get("slug")
    token_ids_str = market_info.get("clobTokenIds", "[]")
    token_ids = json.loads(token_ids_str)
    
    if not token_ids or len(token_ids) < 2:
        return []
    
    # Binary markets have 2 tokens: [YES, NO]
    yes_results = fetch_token_history(token_ids[0], "YES", m_slug, start_ts, end_ts)
    no_results = fetch_token_history(token_ids[1], "NO", m_slug, start_ts, end_ts)
    
    return yes_results + no_results

def main():
    print(f"Gathering BTC 1-day market data from {START_DATE.date()} to {END_DATE.date()}...")
    
    with open(CSV_FILENAME, 'w', newline='') as csvfile:
        fieldnames = ['event_slug', 'market_slug', 'outcome', 'timestamp', 'price', 'datetime_utc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        current_date = START_DATE
        while current_date <= END_DATE:
            month_name = get_month_name(current_date.month)
            day = current_date.day
            slug = f"bitcoin-above-on-{month_name}-{day}"
            
            market_end_dt = datetime(current_date.year, current_date.month, current_date.day, 17, 0, tzinfo=timezone.utc)
            market_start_dt = market_end_dt - timedelta(days=1)
            
            start_ts = market_start_dt.timestamp()
            end_ts = market_end_dt.timestamp()

            event_data = get_event_data(slug)
            if not event_data:
                # Try fallback with year if needed, but the pattern seems stable
                # print(f"  Slug {slug} not found.")
                current_date += timedelta(days=1)
                continue

            markets = event_data.get("markets", [])
            print(f"Processing {slug} ({len(markets)} markets)...")
            
            # Use threading to fetch histories for all markets of the day in parallel
            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_market = {executor.submit(fetch_market_history, m, start_ts, end_ts): m for m in markets}
                for future in as_completed(future_to_market):
                    market_results = future.result()
                    for row in market_results:
                        row['event_slug'] = slug
                        writer.writerow(row)
            
            current_date += timedelta(days=1)
            time.sleep(0.1) # Small delay between days

    print(f"Finished! Data saved to {CSV_FILENAME}")

if __name__ == "__main__":
    main()
