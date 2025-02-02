import os
import numpy as np
import json
import logging
import shutil
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define folders and files
BASE_FOLDER = "/shared_data"                      # The shared volume
TICKERS_MAPPING_FILE = "/shared_data/tickers.json"  # Mapping file expected in shared_data
OUTPUT_FOLDER = "/shared_data/nova_analysis"        # Where to write the final analysis

# Ensure the output folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Failsafe: if tickers.json is not in /shared_data, copy it from /app (assuming it’s there)
if not os.path.exists(TICKERS_MAPPING_FILE):
    try:
        shutil.copy('/app/tickers.json', TICKERS_MAPPING_FILE)
        logging.info("Copied tickers.json from /app to /shared_data")
    except Exception as e:
        logging.error(f"Could not copy tickers.json: {e}")

# Dictionary to store results for each ticker
ticker_results = {}

def format_money(amount):
    """Format a monetary value with a dollar sign, commas, and no decimals."""
    return f"${amount:,.0f}"

def parse_total_spent(total_spent_str):
    """
    Convert a monetary string (e.g., '$30.8K', '$1.2M', or '$564.00') to a float value.
    """
    try:
        value_str = total_spent_str.replace("$", "").strip()
        multiplier = 1
        if "K" in value_str:
            multiplier = 1_000
            value_str = value_str.replace("K", "")
        elif "M" in value_str:
            multiplier = 1_000_000
            value_str = value_str.replace("M", "")
        return float(value_str) * multiplier
    except Exception as e:
        logging.error(f"Error parsing total_spent value '{total_spent_str}': {e}")
        return 0.0

# ----- WEIGHTED OPEN INTEREST SCORING -----
def weighted_open_interest_scoring(calls_oi, puts_oi):
    total_score = 0
    calls_oi_values = np.array(list(calls_oi.values()))
    puts_oi_values = np.array(list(puts_oi.values()))
    mean_calls_oi = np.mean(calls_oi_values) if len(calls_oi_values) > 0 else 0
    mean_puts_oi = np.mean(puts_oi_values) if len(puts_oi_values) > 0 else 0

    # Weight calls
    for date, call_oi in calls_oi.items():
        if call_oi > 0 and mean_calls_oi > 0:
            weight = call_oi / mean_calls_oi
            if weight >= 3:
                total_score += 3
            elif weight >= 2:
                total_score += 2
            elif weight >= 1.5:
                total_score += 1

    # Weight puts
    for date, put_oi in puts_oi.items():
        if put_oi > 0 and mean_puts_oi > 0:
            weight = put_oi / mean_puts_oi
            if weight >= 3:
                total_score -= 3
            elif weight >= 2:
                total_score -= 2
            elif weight >= 1.5:
                total_score -= 1

    return total_score

def analyze_ticker_json(file_path):
    """
    Reads a JSON file and calculates metrics (score, unusual contracts, etc.)
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

    # Extract key sections from the JSON data
    calls_oi = data.get("calls_oi", {})
    puts_oi = data.get("puts_oi", {})
    max_strike_calls = data.get("max_strike_calls", {})
    max_strike_puts = data.get("max_strike_puts", {})
    second_max_strike_calls = data.get("second_max_strike_calls", {})
    second_max_strike_puts = data.get("second_max_strike_puts", {})
    third_max_strike_calls = data.get("third_max_strike_calls", {})
    third_max_strike_puts = data.get("third_max_strike_puts", {})
    current_price = data.get("current_price", 0)
    top_volume_contracts = data.get("top_volume_contracts", [])
    company_name = data.get("company_name", "")

    total_score = 0
    unusual_contracts_count = 0
    total_unusual_spent = 0.0
    cumulative_total_spent_calls = 0.0
    cumulative_total_spent_puts = 0.0

    # ----- Weighted Open Interest Scoring -----
    total_score += weighted_open_interest_scoring(calls_oi, puts_oi)

    # ----- OI Ratio Additions -----
    for date in calls_oi.keys():
        call_val = calls_oi.get(date, 0)
        put_val = puts_oi.get(date, 0)
        if call_val > 0 and put_val > 0:
            if call_val / put_val >= 7:
                total_score += 10
            if put_val / call_val >= 7:
                total_score -= 10

    # ----- Current Price Scoring -----
    percentage_thresholds = [5, 10, 15, 20, 25, 30, 35]
    for date in max_strike_calls.keys():
        max_call = max_strike_calls.get(date, 0)
        max_put = max_strike_puts.get(date, 0)
        second_call = second_max_strike_calls.get(date, 0)
        second_put = second_max_strike_puts.get(date, 0)
        third_call = third_max_strike_calls.get(date, 0)
        third_put = third_max_strike_puts.get(date, 0)

        if current_price < max_call:
            total_score += 1
        if current_price < max_put:
            total_score += 1
        if current_price > max_call:
            total_score -= 1
        if current_price > max_put:
            total_score -= 1

        for pct in percentage_thresholds:
            if current_price * (1 + pct / 100) < max_call:
                total_score += 1
            if current_price * (1 + pct / 100) < max_put:
                total_score += 1
            if current_price * (1 + pct / 100) < second_call:
                total_score += 1
            if current_price * (1 + pct / 100) < second_put:
                total_score += 1
            if current_price * (1 + pct / 100) < third_call:
                total_score += 1
            if current_price * (1 + pct / 100) < third_put:
                total_score += 1

            if current_price * (1 - pct / 100) > max_call:
                total_score -= 1
            if current_price * (1 - pct / 100) > max_put:
                total_score -= 1
            if current_price * (1 - pct / 100) > second_call:
                total_score -= 1
            if current_price * (1 - pct / 100) > second_put:
                total_score -= 1
            if current_price * (1 - pct / 100) > third_call:
                total_score -= 1
            if current_price * (1 - pct / 100) > third_put:
                total_score -= 1

    # ----- Top Volume Contracts Scoring & Metrics -----
    call_unusual_counter = 0
    put_unusual_counter = 0
    call_unusual_bonus = [25, 50, 75, 100, 125, 150, 125, 150, 200, 250, 275, 300]
    put_unusual_bonus = [-25, -50, -75, -100, -125, -150, -125, -150, -200, -250, -275, -300]

    for contract in top_volume_contracts:
        contract_type = contract.get("type", "")
        strike = contract.get("strike", 0)
        volume = contract.get("volume", 0)
        open_interest = contract.get("openInterest", 0) or 0
        unusual = contract.get("unusual", False)
        total_spent_str = contract.get("total_spent", "$0")
        spent = parse_total_spent(total_spent_str)
        strike_pct_diff = ((strike - current_price) / current_price) * 100 if current_price != 0 else 0

        if contract_type == "CALL":
            cumulative_total_spent_calls += spent
        elif contract_type == "PUT":
            cumulative_total_spent_puts += spent

        if unusual:
            unusual_contracts_count += 1
            total_unusual_spent += spent
            if contract_type == "CALL":
                call_unusual_counter += 1
                idx = call_unusual_counter - 1
                bonus = call_unusual_bonus[idx] if idx < len(call_unusual_bonus) else call_unusual_bonus[-1]
                total_score += bonus
            elif contract_type == "PUT":
                put_unusual_counter += 1
                idx = put_unusual_counter - 1
                bonus = put_unusual_bonus[idx] if idx < len(put_unusual_bonus) else put_unusual_bonus[-1]
                total_score += bonus

        if contract_type == "CALL":
            if volume > open_interest:
                total_score += 2
            if strike_pct_diff >= 20:
                total_score += 3
            elif strike_pct_diff >= 15:
                total_score += 2
            elif strike_pct_diff >= 10:
                total_score += 1
            elif strike_pct_diff >= 5:
                total_score += 0.5
            if spent > 500_000:
                total_score += 0.5
            if spent > 1_000_000:
                total_score += 1
            if spent > 5_000_000:
                total_score += 2
            if spent > 10_000_000:
                total_score += 3

        elif contract_type == "PUT":
            if volume > open_interest:
                total_score -= 2
            if strike_pct_diff <= -20:
                total_score -= 3
            elif strike_pct_diff <= -15:
                total_score -= 2
            elif strike_pct_diff <= -10:
                total_score -= 1
            elif strike_pct_diff <= -5:
                total_score -= 0.5
            if spent > 500_000:
                total_score -= 0.5
            if spent > 1_000_000:
                total_score -= 1
            if spent > 5_000_000:
                total_score -= 2
            if spent > 10_000_000:
                total_score -= 3

    if cumulative_total_spent_calls > cumulative_total_spent_puts:
        total_score += 5
    elif cumulative_total_spent_calls < cumulative_total_spent_puts:
        total_score -= 5

    result = {
        "score": total_score,
        "unusual_contracts_count": unusual_contracts_count,
        "total_unusual_spent": total_unusual_spent,  # raw float; will be formatted later
        "cumulative_total_spent_calls": cumulative_total_spent_calls,  # raw float
        "cumulative_total_spent_puts": cumulative_total_spent_puts,      # raw float
        "current_price": current_price,
        "company_name": company_name
    }
    return result

# --- LOAD TICKERS MAPPING & BUILD REVERSE LOOKUP ---
# We build a reverse mapping from ticker symbol to {sector, industry}.
ticker_to_sector_industry = {}
try:
    with open(TICKERS_MAPPING_FILE, "r") as f:
        tickers_mapping = json.load(f)
    for sector, industries in tickers_mapping.items():
        for industry, tickers in industries.items():
            for ticker in tickers:
                # Normalize ticker symbols to uppercase
                ticker_to_sector_industry[ticker.upper()] = {"sector": sector, "industry": industry}
except Exception as e:
    logging.error(f"Error reading {TICKERS_MAPPING_FILE}: {e}")

# --- PROCESS TICKER FOLDERS ---
# (Exclude the output folder to avoid processing non-ticker directories.)
for ticker_folder in os.listdir(BASE_FOLDER):
    # Skip the output directory and any files that aren’t directories
    if ticker_folder == os.path.basename(OUTPUT_FOLDER):
        continue
    folder_path = os.path.join(BASE_FOLDER, ticker_folder)
    if os.path.isdir(folder_path):
        # Normalize folder name (assuming folder names are ticker symbols)
        ticker_symbol = ticker_folder.upper()
        for file in os.listdir(folder_path):
            if file.endswith("_raw.json"):
                file_path = os.path.join(folder_path, file)
                logging.info(f"Processing: {file_path}")
                result = analyze_ticker_json(file_path)
                if result is not None:
                    # Look up the ticker using the normalized symbol; default to Unknown if not found
                    mapping = ticker_to_sector_industry.get(ticker_symbol, {"sector": "Unknown", "industry": "Unknown"})
                    result.update(mapping)
                    ticker_results[ticker_folder] = result

# --- AGGREGATE DATA BY SECTOR & INDUSTRY ---
sector_aggregates = {}
industry_aggregates = {}

for ticker, data in ticker_results.items():
    sector = data.get("sector", "Unknown")
    industry = data.get("industry", "Unknown")

    if sector not in sector_aggregates:
        sector_aggregates[sector] = {
            "total_score": 0,
            "ticker_count": 0,
            "total_unusual_spent": 0,
            "total_calls_spent": 0,
            "total_puts_spent": 0
        }
    sector_aggregates[sector]["total_score"] += data["score"]
    sector_aggregates[sector]["ticker_count"] += 1
    sector_aggregates[sector]["total_unusual_spent"] += data["total_unusual_spent"]
    sector_aggregates[sector]["total_calls_spent"] += data["cumulative_total_spent_calls"]
    sector_aggregates[sector]["total_puts_spent"] += data["cumulative_total_spent_puts"]

    industry_key = f"{sector}|{industry}"
    if industry_key not in industry_aggregates:
        industry_aggregates[industry_key] = {
            "total_score": 0,
            "ticker_count": 0,
            "total_unusual_spent": 0,
            "total_calls_spent": 0,
            "total_puts_spent": 0
        }
    industry_aggregates[industry_key]["total_score"] += data["score"]
    industry_aggregates[industry_key]["ticker_count"] += 1
    industry_aggregates[industry_key]["total_unusual_spent"] += data["total_unusual_spent"]
    industry_aggregates[industry_key]["total_calls_spent"] += data["cumulative_total_spent_calls"]
    industry_aggregates[industry_key]["total_puts_spent"] += data["cumulative_total_spent_puts"]

for sector, agg in sector_aggregates.items():
    agg["average_score"] = agg["total_score"] / agg["ticker_count"] if agg["ticker_count"] > 0 else 0
    agg["formatted_total_unusual_spent"] = format_money(agg["total_unusual_spent"])
    agg["formatted_total_calls_spent"] = format_money(agg["total_calls_spent"])
    agg["formatted_total_puts_spent"] = format_money(agg["total_puts_spent"])

for industry_key, agg in industry_aggregates.items():
    agg["average_score"] = agg["total_score"] / agg["ticker_count"] if agg["ticker_count"] > 0 else 0
    agg["formatted_total_unusual_spent"] = format_money(agg["total_unusual_spent"])
    agg["formatted_total_calls_spent"] = format_money(agg["total_calls_spent"])
    agg["formatted_total_puts_spent"] = format_money(agg["total_puts_spent"])

# --- FORMAT TICKER-LEVEL MONETARY FIELDS ---
for ticker, data in ticker_results.items():
    data["total_unusual_spent"] = format_money(data["total_unusual_spent"])
    data["cumulative_total_spent_calls"] = format_money(data["cumulative_total_spent_calls"])
    data["cumulative_total_spent_puts"] = format_money(data["cumulative_total_spent_puts"])

# --- SORT TICKERS & DETERMINE TOP SECTORS ---
sorted_ticker_results = dict(sorted(ticker_results.items(), key=lambda item: item[1]["score"], reverse=True))
top_20_bullish = dict(list(sorted_ticker_results.items())[:40])
top_20_bearish = dict(list(sorted_ticker_results.items())[-40:])

sorted_sectors = sorted(sector_aggregates.items(), key=lambda item: item[1]["average_score"], reverse=True)
top_bullish_sector = sorted_sectors[0] if sorted_sectors else None
top_bearish_sector = sorted_sectors[-1] if sorted_sectors else None

# --- SAVE THE FINAL RESULTS ---
summary_results = {
    "top_20_bullish": top_20_bullish,
    "top_20_bearish": top_20_bearish,
    "all_tickers": sorted_ticker_results,
    "sector_summary": sector_aggregates,
    "industry_summary": industry_aggregates,
    "top_bullish_sector": top_bullish_sector,
    "top_bearish_sector": top_bearish_sector
}

summary_file = os.path.join(OUTPUT_FOLDER, "summary_results2.json")
with open(summary_file, "w") as outfile:
    json.dump(summary_results, outfile, indent=4)

logging.info("Scoring Complete")
logging.info(f"Results saved to: {summary_file}")
