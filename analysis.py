import os
import numpy as np
import json
import logging
import shutil
import time
from typing import Dict, Any, Optional
import csv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define folders and files
BASE_FOLDER = "/shared_data"                       # The shared volume
TICKERS_MAPPING_FILE = "/shared_data/tickers.json" # Mapping file expected in shared_data
OUTPUT_FOLDER = "/shared_data/nova_analysis"       # Where to write the final analysis

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

def parse_total_spent(total_spent_str: str) -> float:
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
def weighted_open_interest_scoring(calls_oi: Dict[str, float], puts_oi: Dict[str, float]) -> float:
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
    Returns a dictionary of results (score, spent, etc.) for aggregator usage.
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

        # Basic check: current price relative to max_strike_calls / puts
        if current_price < max_call:
            total_score += 0.5
        if current_price < max_put:
            total_score += 1
        if current_price > max_call:
            total_score -= 0.5
        if current_price > max_put:
            total_score -= 1

        # Tiered checks using percentage thresholds
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
    call_unusual_bonus = [20, 40, 70, 85, 110, 130, 145, 175, 200, 250, 275, 300]
    put_unusual_bonus = [-20, -40, -70, -85, -110, -130, -145, -175, -200, -250, -275, -300]

    for contract in top_volume_contracts:
        contract_type = contract.get("type", "")
        strike = contract.get("strike", 0)
        volume = contract.get("volume", 0)
        open_interest = contract.get("openInterest", 0) or 0
        unusual = contract.get("unusual", False)
        total_spent_str = contract.get("total_spent", "$0")
        spent = parse_total_spent(total_spent_str)
        strike_pct_diff = ((strike - current_price) / current_price) * 100 if current_price != 0 else 0

        # Track spent for calls vs. puts
        if contract_type == "CALL":
            cumulative_total_spent_calls += spent
        elif contract_type == "PUT":
            cumulative_total_spent_puts += spent

        # If unusual, update counters + bonus scoring
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

        # Additional call scoring
        if contract_type == "CALL":
            if volume > open_interest:
                total_score += 2.5
            if strike_pct_diff >= 20:
                total_score += 2
            elif strike_pct_diff >= 15:
                total_score += 1.5
            elif strike_pct_diff >= 10:
                total_score += 1
            elif strike_pct_diff >= 5:
                total_score += 0.5

            # Large spend thresholds
            if spent > 250_000:
                total_score += 0.25
            if spent > 500_000:
                total_score += 0.5
            if spent > 1_000_000:
                total_score += 1
            if spent > 2_000_000:
                total_score += 2
            if spent > 2_500_000:
                total_score += 2.5
            if spent > 5_000_000:
                total_score += 3
            if spent > 10_000_000:
                total_score += 5

        # Additional put scoring
        elif contract_type == "PUT":
            if volume > open_interest:
                total_score -= 2.5
            if strike_pct_diff <= -20:
                total_score -= 2
            elif strike_pct_diff <= -15:
                total_score -= 1.5
            elif strike_pct_diff <= -10:
                total_score -= 1
            elif strike_pct_diff <= -5:
                total_score -= 0.5

            # Large spend thresholds
            if spent > 250_000:
                total_score -= 0.25
            if spent > 500_000:
                total_score -= 0.5
            if spent > 1_000_000:
                total_score -= 1
            if spent > 2_000_000:
                total_score -= 2
            if spent > 2_500_000:
                total_score -= 2.5
            if spent > 5_000_000:
                total_score -= 3
            if spent > 10_000_000:
                total_score -= 5

    # Compare total spent calls vs. puts
    if cumulative_total_spent_calls > cumulative_total_spent_puts:
        total_score += 3
    elif cumulative_total_spent_calls < cumulative_total_spent_puts:
        total_score -= 3

    result = {
        "score": total_score,
        "unusual_contracts_count": unusual_contracts_count,
        "total_unusual_spent": total_unusual_spent,
        "cumulative_total_spent_calls": cumulative_total_spent_calls,
        "cumulative_total_spent_puts": cumulative_total_spent_puts,
        "current_price": current_price,
        "company_name": company_name
    }
    return result


# -------------------------
# COMPARISON FUNCTIONS
# -------------------------
def compare_oi_fields(old_data, new_data, csv_path):
    """
    Compare calls_oi and puts_oi, writing changes to CSV.
    """
    calls_old = old_data.get("calls_oi", {})
    calls_new = new_data.get("calls_oi", {})
    puts_old = old_data.get("puts_oi", {})
    puts_new = new_data.get("puts_oi", {})

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category", "Date", "OldValue", "NewValue", "Difference"])

        # Compare calls_oi
        _compare_dict(writer, "calls_oi", calls_old, calls_new)
        # Compare puts_oi
        _compare_dict(writer, "puts_oi", puts_old, puts_new)

def _compare_dict(writer, category, old_dict, new_dict):
    """
    Helper to compare dicts of {date: numeric_value} for calls_oi, puts_oi, etc.
    """
    all_dates = set(old_dict.keys()) | set(new_dict.keys())
    for date in sorted(all_dates):
        old_val = float(old_dict.get(date, 0))
        new_val = float(new_dict.get(date, 0))
        if old_val != new_val:
            diff = new_val - old_val
            writer.writerow([category, date, old_val, new_val, diff])

def compare_strike_fields(old_data, new_data, csv_path):
    """
    Compare max_strike_calls/puts, second_max_strike_calls/puts, third_max_strike_calls/puts.
    """
    fields = [
        "max_strike_calls", "max_strike_puts",
        "second_max_strike_calls", "second_max_strike_puts",
        "third_max_strike_calls", "third_max_strike_puts"
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category", "Date", "OldValue", "NewValue", "Difference"])

        for field in fields:
            old_dict = old_data.get(field, {})
            new_dict = new_data.get(field, {})
            all_dates = set(old_dict.keys()) | set(new_dict.keys())
            for date in sorted(all_dates):
                old_val = float(old_dict.get(date, 0))
                new_val = float(new_dict.get(date, 0))
                if old_val != new_val:
                    diff = new_val - old_val
                    writer.writerow([field, date, old_val, new_val, diff])

def compare_top_volume_contracts(old_data, new_data, csv_path):
    """
    Compare top_volume_contracts. We'll identify each contract by (type, strike, date).
    """
    old_contracts = old_data.get("top_volume_contracts", [])
    new_contracts = new_data.get("top_volume_contracts", [])

    # Build dicts keyed by a signature
    old_dict = {}
    for c in old_contracts:
        sig = contract_signature(c)
        old_dict[sig] = c

    new_dict = {}
    for c in new_contracts:
        sig = contract_signature(c)
        new_dict[sig] = c

    all_sigs = set(old_dict.keys()) | set(new_dict.keys())

    fields_to_compare = ["strike", "volume", "openInterest", "date", "total_spent", "unusual"]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category", "Signature", "FieldName", "OldValue", "NewValue", "Difference"])

        for sig in sorted(all_sigs):
            old_c = old_dict.get(sig, {})
            new_c = new_dict.get(sig, {})

            if not old_c and new_c:
                # brand new contract
                writer.writerow(["top_volume_contracts", sig, "CONTRACT_STATUS", "MISSING", "ADDED", None])
                continue
            elif old_c and not new_c:
                # removed
                writer.writerow(["top_volume_contracts", sig, "CONTRACT_STATUS", "REMOVED", "MISSING", None])
                continue

            # Compare field by field
            for f_name in fields_to_compare:
                old_val = old_c.get(f_name)
                new_val = new_c.get(f_name)
                if old_val == new_val:
                    continue

                diff = compute_difference(f_name, old_val, new_val)
                writer.writerow(["top_volume_contracts", sig, f_name, old_val, new_val, diff])

def contract_signature(contract):
    """
    Unique signature for a contract based on (type, strike, date).
    """
    ctype = contract.get("type", "UNKNOWN")
    strike = contract.get("strike", "UNKNOWN")
    date = contract.get("date", "UNKNOWN")
    return f"{ctype}|{strike}|{date}"

def compute_difference(field_name, old_val, new_val):
    """
    Compute difference for numeric fields, or mark changes for booleans/strings.
    """
    if field_name in ["strike", "volume", "openInterest"]:
        old_num = float(old_val) if is_number(old_val) else 0.0
        new_num = float(new_val) if is_number(new_val) else 0.0
        return new_num - old_num

    if field_name == "total_spent":
        old_num = parse_total_spent(old_val) if old_val else 0.0
        new_num = parse_total_spent(new_val) if new_val else 0.0
        return new_num - old_num

    if field_name == "unusual":
        return 1 if old_val != new_val else 0  # or True/False

    if field_name == "date":
        return "CHANGED" if old_val != new_val else None

    return None

def is_number(val):
    try:
        float(val)
        return True
    except:
        return False

# -------------------------
# BUILD REVERSE LOOKUP
# -------------------------
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

# -------------------------
# PROCESS TICKER FOLDERS
# -------------------------
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

                # -------------------------------------------------------------
                # 1) Compare Old vs. New (raw data) if we have an old.json
                # -------------------------------------------------------------
                old_file_path = os.path.join(folder_path, "old.json")
                if os.path.exists(old_file_path):
                    # Load old/new
                    try:
                        with open(old_file_path, "r") as f:
                            old_data = json.load(f)
                        with open(file_path, "r") as f:
                            new_data = json.load(f)
                        # Compare and write CSVs
                        compare_oi_fields(old_data, new_data, os.path.join(OUTPUT_FOLDER, f"{ticker_symbol}_oi_changes.csv"))
                        compare_strike_fields(old_data, new_data, os.path.join(OUTPUT_FOLDER, f"{ticker_symbol}_strikes_changes.csv"))
                        compare_top_volume_contracts(old_data, new_data, os.path.join(OUTPUT_FOLDER, f"{ticker_symbol}_contracts_changes.csv"))
                    except Exception as e:
                        logging.error(f"Error comparing old vs new data for {ticker_symbol}: {e}")

                # -------------------------------------------------------------
                # 2) Perform the normal analysis scoring
                # -------------------------------------------------------------
                result = analyze_ticker_json(file_path)
                if result is not None:
                    # Look up the ticker using the normalized symbol; default to Unknown if not found
                    mapping = ticker_to_sector_industry.get(ticker_symbol, {"sector": "Unknown", "industry": "Unknown"})
                    result.update(mapping)
                    ticker_results[ticker_folder] = result

                # -------------------------------------------------------------
                # 3) Copy new raw data to old.json for next run
                # -------------------------------------------------------------
                try:
                    shutil.copy(file_path, old_file_path)
                except Exception as e:
                    logging.error(f"Error copying {file_path} to {old_file_path}: {e}")

# -------------------------
# AGGREGATE DATA BY SECTOR & INDUSTRY
# -------------------------
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

# Compute averages and format totals
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

# Format monetary fields in ticker_results
for ticker, data in ticker_results.items():
    data["total_unusual_spent"] = format_money(data["total_unusual_spent"])
    data["cumulative_total_spent_calls"] = format_money(data["cumulative_total_spent_calls"])
    data["cumulative_total_spent_puts"] = format_money(data["cumulative_total_spent_puts"])

# Sort tickers & determine top sectors
sorted_ticker_results = dict(sorted(ticker_results.items(), key=lambda item: item[1]["score"], reverse=True))
top_20_bullish = dict(list(sorted_ticker_results.items())[:40])
top_20_bearish = dict(list(sorted_ticker_results.items())[-40:])

sorted_sectors = sorted(sector_aggregates.items(), key=lambda item: item[1]["average_score"], reverse=True)
top_bullish_sector = sorted_sectors[0] if sorted_sectors else None
top_bearish_sector = sorted_sectors[-1] if sorted_sectors else None

# Save final results
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
