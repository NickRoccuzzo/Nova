import os
import json
import logging
import shutil
import sqlite3
from typing import Dict, Any, Optional


# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define folders and files
BASE_FOLDER = "/shared_data"                       # The shared volume
TICKERS_MAPPING_FILE = "/shared_data/tickers.json" # Mapping file expected in shared_data
OUTPUT_FOLDER = "/shared_data/nova_analysis"       # Where to write the final analysis
DATABASE_FILE = "/shared_data/my_analysis.db"


# Ensure the output folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Failsafe: if tickers.json is not in /shared_data, copy it from /app (assuming it’s there, but it should always be)
if not os.path.exists(TICKERS_MAPPING_FILE):
    try:
        shutil.copy('/app/tickers.json', TICKERS_MAPPING_FILE)
        logging.info("Copied tickers.json from /app to /shared_data")
    except Exception as e:
        logging.error(f"Could not copy tickers.json: {e}")

# Dictionary to store results for each ticker (score, etc.)
ticker_results = {}


# -- SQLITE DATABASE SECTION -- #


def setup_database(db_file: str):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_json_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            json_content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tracked_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            change_type TEXT NOT NULL,
            date_key TEXT,
            old_value TEXT,
            new_value TEXT,
            diff TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def fetch_latest_json_for_ticker(db_file: str, ticker: str):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT json_content
        FROM raw_json_data
        WHERE ticker = ?
        ORDER BY created_at DESC
        LIMIT 1
    ''', (ticker,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]  # json_content
    return None


def insert_raw_json(db_file: str, ticker: str, json_content: str):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO raw_json_data (ticker, json_content)
        VALUES (?, ?)
    ''', (ticker, json_content))
    conn.commit()
    conn.close()


def store_tracked_changes(changes: list):
    """
    Insert each change record into the 'tracked_changes' table.
    """
    if not changes:
        return
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    for change in changes:
        cursor.execute('''
            INSERT INTO tracked_changes 
            (ticker, change_type, date_key, old_value, new_value, diff)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            change["ticker"],
            change["change_type"],
            change.get("date_key", None),
            change.get("old_value", ""),
            change.get("new_value", ""),
            change.get("diff", "")
        ))
    conn.commit()
    conn.close()


def store_raw_data_in_db(file_path: str, ticker: str):
    """
    Reads the new _raw.json, compares to old data if any,
    and stores both the new JSON and the tracked changes.
    """
    # 1) Parse the new file
    try:
        with open(file_path, 'r') as f:
            new_json_str = f.read()
        new_data = json.loads(new_json_str)
    except Exception as e:
        logging.error(f"Could not read or parse {file_path}: {e}")
        return

    # 2) Fetch the old data from the DB
    old_json_str = fetch_latest_json_for_ticker(DATABASE_FILE, ticker)
    if old_json_str:
        old_data = json.loads(old_json_str)

        # 3) Compare old vs. new
        changes = compare_raw_data(old_data, new_data, ticker)

        # 4) Store changes in DB (if any)
        if changes:  # changes will be a list of dictionary rows
            store_tracked_changes(changes)
    else:
        logging.info(f"No old data found for {ticker}, inserting the new JSON as first record.")

    # 5) Insert the new JSON
    insert_raw_json(DATABASE_FILE, ticker, new_json_str)
    logging.info(f"Inserted new raw JSON data for {ticker} into the database.")


# -- COMPARISON LOGIC -- #


def compare_raw_data(old_data: dict, new_data: dict, ticker: str):
    """
    Compare relevant fields in old_data vs. new_data and return
    a list of 'change' dictionaries, each describing the difference.
    """
    changes = []

    # 1) Compare calls_oi, puts_oi, etc. (all dicts keyed by date) for absolute +/- difference
    date_based_fields = [
        "calls_oi", "puts_oi",
        "max_strike_calls", "second_max_strike_calls", "third_max_strike_calls",
        "max_strike_puts", "second_max_strike_puts", "third_max_strike_puts"
    ]

    for field_name in date_based_fields:
        old_dict = old_data.get(field_name, {})
        new_dict = new_data.get(field_name, {})

        # For each date in the new data, compare old vs new
        for date_key, new_val in new_dict.items():
            old_val = old_dict.get(date_key)
            if old_val is not None and old_val != new_val:
                diff_num = new_val - old_val
                # Format difference with sign
                diff_str = f"{'+' if diff_num >= 0 else ''}{diff_num}"

                changes.append({
                    "ticker": ticker,
                    "change_type": field_name,
                    "date_key": date_key,
                    "old_value": str(old_val),
                    "new_value": str(new_val),
                    "diff": diff_str
                })
            elif old_val is None:
                # This date_key is new in the new_data
                changes.append({
                    "ticker": ticker,
                    "change_type": field_name,
                    "date_key": date_key,
                    "old_value": "N/A",
                    "new_value": str(new_val),
                    "diff": "NEW"
                })

        # Also check if something existed in old but not in new
        for date_key in old_dict.keys():
            if date_key not in new_dict:
                old_val = old_dict[date_key]
                changes.append({
                    "ticker": ticker,
                    "change_type": field_name,
                    "date_key": date_key,
                    "old_value": str(old_val),
                    "new_value": "N/A",
                    "diff": "REMOVED"
                })

    # 2) Compare avg_strike with percentage changes
    old_avg = old_data.get("avg_strike", {})
    new_avg = new_data.get("avg_strike", {})
    for date_key, new_val in new_avg.items():
        old_val = old_avg.get(date_key)
        if old_val is not None and old_val != new_val:
            if old_val != 0:
                pct_diff = ((new_val - old_val) / old_val) * 100
            else:
                pct_diff = 999.99  # Arbitrary large if old_val is zero
            diff_str = f"{pct_diff:+.2f}%"
            changes.append({
                "ticker": ticker,
                "change_type": "avg_strike",
                "date_key": date_key,
                "old_value": f"{old_val:.2f}",
                "new_value": f"{new_val:.2f}",
                "diff": diff_str
            })
        elif old_val is None:
            # new date in avg_strike
            changes.append({
                "ticker": ticker,
                "change_type": "avg_strike",
                "date_key": date_key,
                "old_value": "N/A",
                "new_value": f"{new_val:.2f}",
                "diff": "NEW"
            })

    for date_key in old_avg.keys():
        if date_key not in new_avg:
            old_val = old_avg[date_key]
            changes.append({
                "ticker": ticker,
                "change_type": "avg_strike",
                "date_key": date_key,
                "old_value": f"{old_val:.2f}",
                "new_value": "N/A",
                "diff": "REMOVED"
            })

    # 3) Compare top_volume_contracts
    # We'll match "contracts" by a unique combination of (type, date, strike).
    # Then compare volume, openInterest, total_spent, unusual, etc.

    old_contracts = old_data.get("top_volume_contracts", [])
    new_contracts = new_data.get("top_volume_contracts", [])

    # Build a lookup dict for old contracts
    old_lookup = {}
    for c in old_contracts:
        key = contract_signature(c)  # We'll define it below
        old_lookup[key] = c

    # Check new vs old
    for c_new in new_contracts:
        key = contract_signature(c_new)
        c_old = old_lookup.get(key)

        if not c_old:
            # Entirely new contract
            changes.append({
                "ticker": ticker,
                "change_type": "top_volume_contracts",
                "date_key": c_new["date"],
                "old_value": "N/A",
                "new_value": f"{c_new}",
                "diff": "NEW CONTRACT"
            })
        else:
            # Compare fields: strike, volume, openInterest, total_spent, unusual
            # If strike changed (rare?), do a % diff
            if c_old["strike"] != c_new["strike"]:
                strike_old = c_old["strike"]
                strike_new = c_new["strike"]
                if strike_old != 0:
                    strike_diff_pct = ((strike_new - strike_old) / strike_old) * 100
                else:
                    strike_diff_pct = 999.99
                changes.append({
                    "ticker": ticker,
                    "change_type": "top_volume_contracts.strike",
                    "date_key": c_new["date"],
                    "old_value": str(strike_old),
                    "new_value": str(strike_new),
                    "diff": f"{strike_diff_pct:+.2f}%"
                })

            # Compare volume (absolute difference)
            if c_old["volume"] != c_new["volume"]:
                vol_diff = c_new["volume"] - c_old["volume"]
                changes.append({
                    "ticker": ticker,
                    "change_type": "top_volume_contracts.volume",
                    "date_key": c_new["date"],
                    "old_value": str(c_old["volume"]),
                    "new_value": str(c_new["volume"]),
                    "diff": f"{vol_diff:+}"
                })

            # Compare openInterest (absolute difference)
            if c_old["openInterest"] != c_new["openInterest"]:
                oi_diff = c_new["openInterest"] - c_old["openInterest"]
                changes.append({
                    "ticker": ticker,
                    "change_type": "top_volume_contracts.openInterest",
                    "date_key": c_new["date"],
                    "old_value": str(c_old["openInterest"]),
                    "new_value": str(c_new["openInterest"]),
                    "diff": f"{oi_diff:+}"
                })

            # Compare total_spent -> Need to parse it to compare numeric values
            if c_old["total_spent"] != c_new["total_spent"]:
                # Optionally parse to float for real differences
                old_spent = parse_total_spent(c_old["total_spent"])
                new_spent = parse_total_spent(c_new["total_spent"])
                diff_spent = new_spent - old_spent
                diff_spent_str = f"{diff_spent:+.2f}"
                changes.append({
                    "ticker": ticker,
                    "change_type": "top_volume_contracts.total_spent",
                    "date_key": c_new["date"],
                    "old_value": c_old["total_spent"],
                    "new_value": c_new["total_spent"],
                    "diff": diff_spent_str
                })

            # Compare unusual
            if c_old["unusual"] != c_new["unusual"]:
                changes.append({
                    "ticker": ticker,
                    "change_type": "top_volume_contracts.unusual",
                    "date_key": c_new["date"],
                    "old_value": str(c_old["unusual"]),
                    "new_value": str(c_new["unusual"]),
                    "diff": "CHANGED"
                })

    # Check for removed contracts (in old but not in new)
    new_lookup = {contract_signature(c): c for c in new_contracts}
    for c_old in old_contracts:
        key = contract_signature(c_old)
        if key not in new_lookup:
            changes.append({
                "ticker": ticker,
                "change_type": "top_volume_contracts",
                "date_key": c_old["date"],
                "old_value": f"{c_old}",
                "new_value": "N/A",
                "diff": "REMOVED CONTRACT"
            })

    return changes


def contract_signature(contract: dict) -> str:
    """Create a unique signature for a contract."""
    return f"{contract.get('type')}|{contract.get('date')}|{contract.get('strike')}"


# -- ANALYSIS LOGIC -- #


def parse_total_spent(total_spent_str: str) -> float:
    """A quick parse function to convert strings like '$5.0M' to float(5000000)."""
    # You can adapt your existing parse_total_spent logic here
    import re
    import math

    val = total_spent_str.strip().replace('$', '').replace(',', '')

    # Check for suffix
    multiplier = 1.0
    if val.endswith('K'):
        multiplier = 1_000
        val = val[:-1]
    elif val.endswith('M'):
        multiplier = 1_000_000
        val = val[:-1]
    elif val.endswith('B'):
        multiplier = 1_000_000_000
        val = val[:-1]

    try:
        return float(val) * multiplier
    except:
        return 0.0


def format_money(amount):
    """
    Format a monetary value as a string with a dollar sign, commas,
    and:
      - .0 if it's effectively an integer (e.g. 695135.0 -> "$695,135.0")
      - .2f if it has a fractional component (e.g. 28.47 -> "$28.47")
    """

    try:
        # Convert non-numeric inputs via parse_total_spent if needed
        if not isinstance(amount, (int, float)):
            amount = parse_total_spent(str(amount))

        # Check if it's effectively an integer (allowing tiny floating precision errors)
        if abs(amount - round(amount)) < 1e-9:
            # Format with 1 decimal place -> .0
            return f"${amount:,.1f}"
        else:
            # Otherwise format with two decimals
            return f"${amount:,.2f}"
    except (ValueError, TypeError):
        return "Invalid Amount"


def weighted_open_interest_scoring(calls_oi: Dict[str, float], puts_oi: Dict[str, float]) -> float:
    import numpy as np
    total_score = 0
    calls_oi_values = np.array(list(calls_oi.values()))
    puts_oi_values = np.array(list(puts_oi.values()))
    calls_volume_values = np.array(list(calls_volume.values()))
    puts_volume_values = np.array(list(puts_volume.values()))
    mean_calls_oi = np.mean(calls_oi_values) if len(calls_oi_values) > 0 else 0
    mean_puts_oi = np.mean(puts_oi_values) if len(puts_oi_values) > 0 else 0
    mean_calls_volume = np.mean(calls_volume_values) if len(calls_volume_values) > 0 else 0
    mean_puts_volume = np.mean(puts_volume_values) if len(puts_volume_values) > 0 else 0

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


# -- THE 'MAIN' SECTION -- #


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
# First, ensure DB is set up
setup_database(DATABASE_FILE)

for ticker_folder in os.listdir(BASE_FOLDER):
    if ticker_folder == os.path.basename(OUTPUT_FOLDER):
        continue
    folder_path = os.path.join(BASE_FOLDER, ticker_folder)
    if os.path.isdir(folder_path):
        ticker_symbol = ticker_folder.upper()

        for file in os.listdir(folder_path):
            if file.endswith("_raw.json"):
                file_path = os.path.join(folder_path, file)
                logging.info(f"Processing: {file_path}")

                # 1) Analyze the ticker to get scoring and metrics
                analysis_result = analyze_ticker_json(file_path)
                if analysis_result:
                    # Put the result in ticker_results under the symbol
                    ticker_results[ticker_symbol] = analysis_result

                # 2) Store the raw data in the DB, which also compares old/new
                store_raw_data_in_db(file_path, ticker_symbol)


# OPTIONAL: Sort by score if you want
sorted_ticker_results = dict(
    sorted(ticker_results.items(), key=lambda item: item[1]["score"], reverse=True)
)

# Format monetary fields as requested
for ticker, data in sorted_ticker_results.items():
    data["total_unusual_spent"] = format_money(data["total_unusual_spent"])
    data["cumulative_total_spent_calls"] = format_money(data["cumulative_total_spent_calls"])
    data["cumulative_total_spent_puts"] = format_money(data["cumulative_total_spent_puts"])
    data["current_price"] = format_money(data["current_price"])

summary_results = {
    "all_tickers": sorted_ticker_results
}

summary_file = os.path.join(OUTPUT_FOLDER, "summary_results2.json")
with open(summary_file, "w") as outfile:
    json.dump(summary_results, outfile, indent=4)

logging.info("Scoring Complete")
logging.info(f"Results saved to: {summary_file}")

# -- Export tracked changes to JSON (tracked_changes) -- #
def export_tracked_changes_to_json(db_file: str, output_folder: str):
    """
    Query the 'tracked_changes' table from the provided SQLite database
    and write it to 'tracked_changes.json' in the output folder.
    """
    import sqlite3  # re-import here for clarity, though it's at top too
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    tracked_changes_data = []
    try:
        cursor.execute("SELECT * FROM tracked_changes")
        rows = cursor.fetchall()

        # Grab the column names (id, ticker, change_type, date_key, old_value, new_value, diff, created_at)
        col_names = [desc[0] for desc in cursor.description]

        # Build list of dicts for JSON serialization
        for row in rows:
            row_dict = dict(zip(col_names, row))
            tracked_changes_data.append(row_dict)

    except sqlite3.Error as e:
        logging.error(f"Database error while exporting tracked_changes: {e}")
    finally:
        conn.close()

    # Construct the output JSON path
    output_json_path = os.path.join(output_folder, "tracked_changes.json")

    # Write the list of dicts to JSON
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(tracked_changes_data, f, indent=2)
        logging.info(f"tracked_changes.json saved to {output_json_path}")
    except Exception as e:
        logging.error(f"Failed to write tracked_changes.json: {e}")


export_tracked_changes_to_json(DATABASE_FILE, OUTPUT_FOLDER)
