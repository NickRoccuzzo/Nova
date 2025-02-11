import os
import numpy as np
import json
import logging
import shutil
import time
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define folders and files
BASE_FOLDER = "/shared_data"                       # The shared volume
TICKERS_MAPPING_FILE = "/shared_data/tickers.json" # Mapping file expected in shared_data
OUTPUT_FOLDER = "/shared_data/nova_analysis"       # Where to write the final analysis
TRACKED_CHANGES_FILE = os.path.join(OUTPUT_FOLDER, "tracked_changes.json")  # Consolidated changes

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

# Dictionary to store results for each ticker (score, etc.)
ticker_results = {}
# Dictionary to store all changes for every ticker
tracked_changes = {}

def parse_total_spent(total_spent_str: str) -> float:
    """
    Convert a monetary string (e.g., '$30.8K', '$1.2M', or '$564.00') to a float value.
    """
    try:
        value_str = total_spent_str.replace("$", "").replace(",", "").strip()
        multiplier = 1

        if value_str.endswith("K"):
            multiplier = 1_000
            value_str = value_str[:-1]  # Remove 'K'
        elif value_str.endswith("M"):
            multiplier = 1_000_000
            value_str = value_str[:-1]  # Remove 'M'
        elif value_str.endswith("B"):
            multiplier = 1_000_000_000
            value_str = value_str[:-1]  # Remove 'B'

        return float(value_str) * multiplier  # Convert to float and apply multiplier
    except Exception as e:
        logging.error(f"Error parsing total_spent value '{total_spent_str}': {e}")
        return 0.0  # Return 0 if parsing fails

def format_money(amount):
    """
    Format a monetary value as a string with a dollar sign, commas, and no decimals.
    This function assumes the input is already a float or int.
    """
    try:
        if not isinstance(amount, (int, float)):
            amount = parse_total_spent(str(amount))  # Convert before formatting

        return f"${amount:,.0f}"  # Format with commas and dollar sign
    except (ValueError, TypeError):
        return "Invalid Amount"  # Handle failed conversions


# ----- WEIGHTED OPEN INTEREST SCORING -----
def weighted_open_interest_scoring(calls_oi: Dict[str, float], puts_oi: Dict[str, float]) -> float:
    import numpy as np
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

    # Weighted Open Interest Scoring
    total_score += weighted_open_interest_scoring(calls_oi, puts_oi)

    # OI Ratio Additions
    for date in calls_oi.keys():
        call_val = calls_oi.get(date, 0)
        put_val = puts_oi.get(date, 0)
        if call_val > 0 and put_val > 0:
            if call_val / put_val >= 7:
                total_score += 10
            if put_val / call_val >= 7:
                total_score -= 10

    # Current Price Scoring
    percentage_thresholds = [5, 10, 15, 20, 25, 30, 35]
    for date in max_strike_calls.keys():
        max_call = max_strike_calls.get(date, 0)
        max_put = max_strike_puts.get(date, 0)
        second_call = second_max_strike_calls.get(date, 0)
        second_put = second_max_strike_puts.get(date, 0)
        third_call = third_max_strike_calls.get(date, 0)
        third_put = third_max_strike_puts.get(date, 0)

        if current_price < max_call:
            total_score += 0.5
        if current_price < max_put:
            total_score += 1
        if current_price > max_call:
            total_score -= 0.5
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

    # Top Volume Contracts Scoring & Metrics
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
                total_score += 2.5
            if strike_pct_diff >= 20:
                total_score += 2
            elif strike_pct_diff >= 15:
                total_score += 1.5
            elif strike_pct_diff >= 10:
                total_score += 1
            elif strike_pct_diff >= 5:
                total_score += 0.5
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


# ---------------
# COMPARISON LOGIC RETURNING DICTS (not CSV)
# ---------------
def compare_oi_fields(old_data, new_data):
    """
    Compare calls_oi and puts_oi, returning a dict of changes:
    {
      "calls_oi": [
         {"date": "02/07/25", "old": 7157, "new": 8157, "diff": 1000},
         ...
      ],
      "puts_oi": [...]
    }
    """
    changes = {
        "calls_oi": [],
        "puts_oi": []
    }
    calls_old = old_data.get("calls_oi", {})
    calls_new = new_data.get("calls_oi", {})
    puts_old = old_data.get("puts_oi", {})
    puts_new = new_data.get("puts_oi", {})

    # calls_oi
    all_dates = set(calls_old.keys()) | set(calls_new.keys())
    for d in sorted(all_dates):
        old_val = float(calls_old.get(d, 0))
        new_val = float(calls_new.get(d, 0))
        if old_val != new_val:
            changes["calls_oi"].append({
                "date": d,
                "old": old_val,
                "new": new_val,
                "diff": new_val - old_val
            })

    # puts_oi
    all_dates = set(puts_old.keys()) | set(puts_new.keys())
    for d in sorted(all_dates):
        old_val = float(puts_old.get(d, 0))
        new_val = float(puts_new.get(d, 0))
        if old_val != new_val:
            changes["puts_oi"].append({
                "date": d,
                "old": old_val,
                "new": new_val,
                "diff": new_val - old_val
            })

    return changes

def compare_strike_fields(old_data, new_data):
    """
    Compare max_strike_calls/puts, second_max_strike_calls/puts, third_max_strike_calls/puts
    returning a dict of changes like:
    {
      "max_strike_calls": [ ... ],
      "max_strike_puts": [ ... ],
      ...
    }
    """
    fields = [
        "max_strike_calls", "max_strike_puts",
        "second_max_strike_calls", "second_max_strike_puts",
        "third_max_strike_calls", "third_max_strike_puts"
    ]
    changes = {}

    for field in fields:
        old_dict = old_data.get(field, {})
        new_dict = new_data.get(field, {})
        field_changes = []
        all_dates = set(old_dict.keys()) | set(new_dict.keys())
        for d in sorted(all_dates):
            old_val = float(old_dict.get(d, 0))
            new_val = float(new_dict.get(d, 0))
            if old_val != new_val:
                field_changes.append({
                    "date": d,
                    "old": old_val,
                    "new": new_val,
                    "diff": new_val - old_val
                })
        changes[field] = field_changes

    return changes

def compare_top_volume_contracts(old_data, new_data):
    """
    Compare top_volume_contracts. Returns a dict like:
    {
      "top_volume_contracts": [
         {
            "signature": "CALL|38.0|02/07/25",
            "field": "volume",
            "old": 498.0,
            "new": 600.0,
            "diff": 102.0
         },
         { "signature": ..., "field": "CONTRACT_STATUS", "old": "MISSING", "new": "ADDED" },
         ...
      ]
    }
    """
    old_contracts = old_data.get("top_volume_contracts", [])
    new_contracts = new_data.get("top_volume_contracts", [])
    changes = {"top_volume_contracts": []}

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

    for sig in sorted(all_sigs):
        old_c = old_dict.get(sig, {})
        new_c = new_dict.get(sig, {})

        if not old_c and new_c:
            # brand new contract
            changes["top_volume_contracts"].append({
                "signature": sig,
                "field": "CONTRACT_STATUS",
                "old": "MISSING",
                "new": "ADDED",
                "diff": None
            })
            continue
        elif old_c and not new_c:
            # removed contract
            changes["top_volume_contracts"].append({
                "signature": sig,
                "field": "CONTRACT_STATUS",
                "old": "REMOVED",
                "new": "MISSING",
                "diff": None
            })
            continue

        # Compare field by field
        for f_name in fields_to_compare:
            old_val = old_c.get(f_name)
            new_val = new_c.get(f_name)
            if old_val == new_val:
                continue
            diff = compute_difference(f_name, old_val, new_val)
            changes["top_volume_contracts"].append({
                "signature": sig,
                "field": f_name,
                "old": old_val,
                "new": new_val,
                "diff": diff
            })

    return changes

def contract_signature(contract):
    """
    Unique signature for a contract based on (type, strike, date).
    """
    ctype = contract.get("type", "UNKNOWN")
    strike = contract.get("strike", "UNKNOWN")
    date = contract.get("date", "UNKNOWN")
    return f"{ctype}|{strike}|{date}"

def compute_difference(field_name, old_val, new_val):
    if field_name in ["strike", "volume", "openInterest"]:
        old_num = float(old_val) if is_number(old_val) else 0.0
        new_num = float(new_val) if is_number(new_val) else 0.0
        return new_num - old_num
    if field_name == "total_spent":
        old_num = parse_total_spent(old_val) if old_val else 0.0
        new_num = parse_total_spent(new_val) if new_val else 0.0
        return new_num - old_num
    if field_name == "unusual":
        return 1 if old_val != new_val else 0
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
                    try:
                        with open(old_file_path, "r") as f:
                            old_data = json.load(f)
                        with open(file_path, "r") as f:
                            new_data = json.load(f)

                        # We'll gather changes in a single dict for this ticker
                        changes_for_ticker = {}

                        # calls_oi/puts_oi changes
                        changes_for_ticker["oi_changes"] = compare_oi_fields(old_data, new_data)

                        # max_strike calls/puts (and second, third)
                        changes_for_ticker["strike_changes"] = compare_strike_fields(old_data, new_data)

                        # top_volume_contracts changes
                        changes_for_ticker["contracts_changes"] = compare_top_volume_contracts(old_data, new_data)

                        # Now put these in the global "tracked_changes" under the ticker symbol
                        tracked_changes[ticker_symbol] = changes_for_ticker

                    except Exception as e:
                        logging.error(f"Error comparing old vs new data for {ticker_symbol}: {e}")

                # -------------------------------------------------------------
                # 2) Perform the normal analysis scoring
                # -------------------------------------------------------------
                result = analyze_ticker_json(file_path)
                if result is not None:
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


# Optionally still sort the tickers by score (or any field):
sorted_ticker_results = dict(
    sorted(ticker_results.items(), key=lambda item: item[1]["score"], reverse=True)
)

summary_results = {
    "all_tickers": sorted_ticker_results
}

summary_file = os.path.join(OUTPUT_FOLDER, "summary_results2.json")
with open(summary_file, "w") as outfile:
    json.dump(summary_results, outfile, indent=4)

logging.info("Scoring Complete")
logging.info(f"Results saved to: {summary_file}")
