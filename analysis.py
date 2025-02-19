import os
import json
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define folders and files
BASE_FOLDER = Path("/shared_data")
TICKERS_MAPPING_FILE = BASE_FOLDER / "tickers.json"  # Mapping file
OUTPUT_FOLDER = BASE_FOLDER / "nova_analysis"         # Analysis output folder
DATABASE_FILE = BASE_FOLDER / "my_analysis.db"         # SQLite DB file

# Ensure the output folder exists
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Failsafe: Copy tickers.json from /app if not present
if not TICKERS_MAPPING_FILE.exists():
    try:
        shutil.copy('/app/tickers.json', TICKERS_MAPPING_FILE)
        logging.info("Copied tickers.json from /app to /shared_data")
    except Exception as e:
        logging.error(f"Could not copy tickers.json: {e}", exc_info=True)

# Dictionary to store results for each ticker
ticker_results: Dict[str, Any] = {}


# -- SQLITE DATABASE SECTION -- #

def setup_database(db_file: Path) -> None:
    conn = sqlite3.connect(str(db_file))
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


def fetch_latest_json_for_ticker(db_file: Path, ticker: str) -> Optional[str]:
    conn = sqlite3.connect(str(db_file))
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
    return row[0] if row else None


def insert_raw_json(db_file: Path, ticker: str, json_content: str) -> None:
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO raw_json_data (ticker, json_content)
        VALUES (?, ?)
    ''', (ticker, json_content))
    conn.commit()
    conn.close()


def store_tracked_changes(changes: List[Dict[str, str]]) -> None:
    if not changes:
        return
    conn = sqlite3.connect(str(DATABASE_FILE))
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


def store_raw_data_in_db(file_path: Path, ticker: str) -> None:
    try:
        new_json_str = file_path.read_text(encoding="utf-8")
        new_data = json.loads(new_json_str)
    except Exception as e:
        logging.error(f"Could not read or parse {file_path}: {e}", exc_info=True)
        return

    old_json_str = fetch_latest_json_for_ticker(DATABASE_FILE, ticker)
    if old_json_str:
        try:
            old_data = json.loads(old_json_str)
            changes = compare_raw_data(old_data, new_data, ticker)
            if changes:
                store_tracked_changes(changes)
        except Exception as e:
            logging.error(f"Error comparing data for {ticker}: {e}", exc_info=True)
    else:
        logging.info(f"No old data found for {ticker}, inserting the new JSON as first record.")

    insert_raw_json(DATABASE_FILE, ticker, new_json_str)
    logging.info(f"Inserted new raw JSON data for {ticker} into the database.")


# -- COMPARISON LOGIC -- #

def compare_dict_fields(field_name: str, old_dict: Dict[str, Any], new_dict: Dict[str, Any], ticker: str) -> List[Dict[str, str]]:
    changes = []
    for date_key, new_val in new_dict.items():
        old_val = old_dict.get(date_key)
        if old_val is not None and old_val != new_val:
            diff_num = new_val - old_val
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
            changes.append({
                "ticker": ticker,
                "change_type": field_name,
                "date_key": date_key,
                "old_value": "N/A",
                "new_value": str(new_val),
                "diff": "NEW"
            })
    for date_key in old_dict.keys():
        if date_key not in new_dict:
            changes.append({
                "ticker": ticker,
                "change_type": field_name,
                "date_key": date_key,
                "old_value": str(old_dict[date_key]),
                "new_value": "N/A",
                "diff": "REMOVED"
            })
    return changes


def compare_avg_strike(old_avg: Dict[str, float], new_avg: Dict[str, float], ticker: str) -> List[Dict[str, str]]:
    changes = []
    for date_key, new_val in new_avg.items():
        old_val = old_avg.get(date_key)
        if old_val is not None and old_val != new_val:
            pct_diff = ((new_val - old_val) / old_val * 100) if old_val != 0 else 999.99
            changes.append({
                "ticker": ticker,
                "change_type": "avg_strike",
                "date_key": date_key,
                "old_value": f"{old_val:.2f}",
                "new_value": f"{new_val:.2f}",
                "diff": f"{pct_diff:+.2f}%"
            })
        elif old_val is None:
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
            changes.append({
                "ticker": ticker,
                "change_type": "avg_strike",
                "date_key": date_key,
                "old_value": f"{old_avg[date_key]:.2f}",
                "new_value": "N/A",
                "diff": "REMOVED"
            })
    return changes


def contract_signature(contract: dict) -> str:
    """Create a unique signature for a contract based on type, date, and strike."""
    return f"{contract.get('type')}|{contract.get('date')}|{contract.get('strike')}"


def parse_total_spent(spent_str: str) -> float:
    """
    Parse a formatted dollar string (e.g., '$18.4K') into a float.
    """
    spent_str = spent_str.replace('$', '').strip()
    multiplier = 1.0
    if spent_str.endswith('B'):
        multiplier = 1_000_000_000
        spent_str = spent_str[:-1]
    elif spent_str.endswith('M'):
        multiplier = 1_000_000
        spent_str = spent_str[:-1]
    elif spent_str.endswith('K'):
        multiplier = 1_000
        spent_str = spent_str[:-1]
    try:
        return float(spent_str) * multiplier
    except ValueError:
        return 0.0


def compare_top_volume_contracts(old_contracts: List[Dict[str, Any]], new_contracts: List[Dict[str, Any]], ticker: str) -> List[Dict[str, str]]:
    changes = []
    old_lookup = {contract_signature(c): c for c in old_contracts}
    new_lookup = {contract_signature(c): c for c in new_contracts}

    for c_new in new_contracts:
        key = contract_signature(c_new)
        c_old = old_lookup.get(key)
        if not c_old:
            changes.append({
                "ticker": ticker,
                "change_type": "top_volume_contracts",
                "date_key": c_new["date"],
                "old_value": "N/A",
                "new_value": f"{c_new}",
                "diff": "NEW CONTRACT"
            })
        else:
            if c_old["strike"] != c_new["strike"]:
                strike_old, strike_new = c_old["strike"], c_new["strike"]
                strike_diff_pct = ((strike_new - strike_old) / strike_old * 100) if strike_old != 0 else 999.99
                changes.append({
                    "ticker": ticker,
                    "change_type": "top_volume_contracts.strike",
                    "date_key": c_new["date"],
                    "old_value": str(strike_old),
                    "new_value": str(strike_new),
                    "diff": f"{strike_diff_pct:+.2f}%"
                })
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
            if c_old["total_spent"] != c_new["total_spent"]:
                old_spent = parse_total_spent(c_old["total_spent"])
                new_spent = parse_total_spent(c_new["total_spent"])
                diff_spent = new_spent - old_spent
                changes.append({
                    "ticker": ticker,
                    "change_type": "top_volume_contracts.total_spent",
                    "date_key": c_new["date"],
                    "old_value": c_old["total_spent"],
                    "new_value": c_new["total_spent"],
                    "diff": f"{diff_spent:+.2f}"
                })
            if c_old["unusual"] != c_new["unusual"]:
                changes.append({
                    "ticker": ticker,
                    "change_type": "top_volume_contracts.unusual",
                    "date_key": c_new["date"],
                    "old_value": str(c_old["unusual"]),
                    "new_value": str(c_new["unusual"]),
                    "diff": "CHANGED"
                })
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


def compare_raw_data(old_data: dict, new_data: dict, ticker: str) -> List[Dict[str, str]]:
    """
    Compare relevant sections of old_data vs. new_data and return a list of differences.
    """
    changes = []
    date_based_fields = [
        "calls_oi", "puts_oi",
        "max_strike_calls", "second_max_strike_calls", "third_max_strike_calls",
        "max_strike_puts", "second_max_strike_puts", "third_max_strike_puts"
    ]
    for field in date_based_fields:
        changes.extend(compare_dict_fields(field, old_data.get(field, {}), new_data.get(field, {}), ticker))
    changes.extend(compare_avg_strike(old_data.get("avg_strike", {}), new_data.get("avg_strike", {}), ticker))
    changes.extend(compare_top_volume_contracts(old_data.get("top_volume_contracts", []), new_data.get("top_volume_contracts", []), ticker))
    return changes


# -- ANALYSIS LOGIC -- #

def format_money(amount: float) -> str:
    """
    Format a monetary value with dollar sign and appropriate commas/decimals.
    """
    try:
        if abs(amount - round(amount)) < 1e-9:
            return f"${amount:,.1f}"
        else:
            return f"${amount:,.2f}"
    except (ValueError, TypeError):
        return "Invalid Amount"


def weighted_open_interest_scoring(calls_oi: Dict[str, float], puts_oi: Dict[str, float]) -> float:
    total_score = 0.0
    calls_values = np.array(list(calls_oi.values()))
    puts_values = np.array(list(puts_oi.values()))
    mean_calls = np.mean(calls_values) if len(calls_values) > 0 else 0
    mean_puts = np.mean(puts_values) if len(puts_values) > 0 else 0

    for date, call_oi in calls_oi.items():
        if call_oi > 0 and mean_calls > 0:
            weight = call_oi / mean_calls
            if weight >= 3:
                total_score += 3
            elif weight >= 2:
                total_score += 2
            elif weight >= 1.5:
                total_score += 1

    for date, put_oi in puts_oi.items():
        if put_oi > 0 and mean_puts > 0:
            weight = put_oi / mean_puts
            if weight >= 3:
                total_score -= 3
            elif weight >= 2:
                total_score -= 2
            elif weight >= 1.5:
                total_score -= 1

    return total_score


def analyze_ticker_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Analyze a ticker's raw JSON file to calculate its score and metrics,
    and include volume data.
    """
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}", exc_info=True)
        return None

    # Extract key sections from the JSON data
    calls_oi = data.get("calls_oi", {})
    puts_oi = data.get("puts_oi", {})
    calls_volume = data.get("calls_volume", {})       # New extraction
    puts_volume = data.get("puts_volume", {})         # New extraction
    max_strike_calls = data.get("max_strike_calls", {})
    max_strike_puts = data.get("max_strike_puts", {})
    second_max_strike_calls = data.get("second_max_strike_calls", {})
    second_max_strike_puts = data.get("second_max_strike_puts", {})
    third_max_strike_calls = data.get("third_max_strike_calls", {})
    third_max_strike_puts = data.get("third_max_strike_puts", {})
    current_price = data.get("current_price", 0)
    top_volume_contracts = data.get("top_volume_contracts", [])
    company_name = data.get("company_name", "")

    total_score = 0.0
    unusual_contracts_count = 0
    total_unusual_spent = 0.0
    cumulative_total_spent_calls = 0.0
    cumulative_total_spent_puts = 0.0

    # Weighted Open Interest Scoring
    total_score += weighted_open_interest_scoring(calls_oi, puts_oi)

    # OI Ratio Adjustments
    for date in calls_oi.keys():
        call_val = calls_oi.get(date, 0)
        put_val = puts_oi.get(date, 0)
        if call_val > 0 and put_val > 0:
            if call_val / put_val >= 7:
                total_score += 10
            if put_val / call_val >= 7:
                total_score -= 10

    # Current Price Scoring using percentage thresholds
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
    call_unusual_bonus = [10, 20, 30, 50, 65, 90, 115, 140, 175, 200, 220, 250]
    put_unusual_bonus = [-10, -20, -30, -50, -65, -90, -115, -140, -175, -200, -220, -250]

    for contract in top_volume_contracts:
        contract_type = contract.get("type", "")
        strike = contract.get("strike", 0)
        volume = contract.get("volume", 0)
        open_interest = contract.get("openInterest", 0) or 0
        unusual = contract.get("unusual", False)
        total_spent_str = contract.get("total_spent", "$0")
        spent = parse_total_spent(total_spent_str)
        strike_pct_diff = ((strike - current_price) / current_price * 100) if current_price != 0 else 0

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

    return {
        "score": total_score,
        "unusual_contracts_count": unusual_contracts_count,
        "total_unusual_spent": total_unusual_spent,
        "cumulative_total_spent_calls": cumulative_total_spent_calls,
        "cumulative_total_spent_puts": cumulative_total_spent_puts,
        "current_price": current_price,
        "company_name": company_name,
        "calls_volume": calls_volume,   # Included in final result
        "puts_volume": puts_volume      # Included in final result
    }


# -- THE 'MAIN' SECTION -- #

# Build reverse lookup from tickers.json
ticker_to_sector_industry: Dict[str, Dict[str, str]] = {}
try:
    tickers_mapping = json.loads(TICKERS_MAPPING_FILE.read_text(encoding="utf-8"))
    for sector, industries in tickers_mapping.items():
        for industry, tickers in industries.items():
            for ticker in tickers:
                ticker_to_sector_industry[ticker.upper()] = {"sector": sector, "industry": industry}
except Exception as e:
    logging.error(f"Error reading {TICKERS_MAPPING_FILE}: {e}", exc_info=True)

# Ensure the database is set up
setup_database(DATABASE_FILE)

# Process each ticker folder in the shared data directory
for ticker_folder in os.listdir(BASE_FOLDER):
    if ticker_folder == OUTPUT_FOLDER.name:
        continue
    folder_path = BASE_FOLDER / ticker_folder
    if folder_path.is_dir():
        ticker_symbol = ticker_folder.upper()
        for file in os.listdir(folder_path):
            if file.endswith("_raw.json"):
                file_path = folder_path / file
                logging.info(f"Processing: {file_path}")
                analysis_result = analyze_ticker_json(file_path)
                if analysis_result:
                    ticker_results[ticker_symbol] = analysis_result
                store_raw_data_in_db(file_path, ticker_symbol)

# Optional: Sort tickers by score (descending)
sorted_ticker_results = dict(
    sorted(ticker_results.items(), key=lambda item: item[1]["score"], reverse=True)
)

# Format monetary fields
for ticker, data in sorted_ticker_results.items():
    data["total_unusual_spent"] = format_money(data["total_unusual_spent"])
    data["cumulative_total_spent_calls"] = format_money(data["cumulative_total_spent_calls"])
    data["cumulative_total_spent_puts"] = format_money(data["cumulative_total_spent_puts"])
    data["current_price"] = format_money(data["current_price"])

summary_results = {"all_tickers": sorted_ticker_results}
summary_file = OUTPUT_FOLDER / "summary_results2.json"
try:
    summary_file.write_text(json.dumps(summary_results, indent=4), encoding="utf-8")
    logging.info(f"Scoring Complete. Results saved to: {summary_file}")
except Exception as e:
    logging.error(f"Failed to write summary results: {e}", exc_info=True)


def export_tracked_changes_to_json(db_file: Path, output_folder: Path) -> None:
    """
    Query the 'tracked_changes' table and export the results to tracked_changes.json.
    """
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    tracked_changes_data = []
    try:
        cursor.execute("SELECT * FROM tracked_changes")
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        for row in rows:
            tracked_changes_data.append(dict(zip(col_names, row)))
    except sqlite3.Error as e:
        logging.error(f"Database error while exporting tracked_changes: {e}", exc_info=True)
    finally:
        conn.close()

    output_json_path = output_folder / "tracked_changes.json"
    try:
        output_json_path.write_text(json.dumps(tracked_changes_data, indent=2), encoding="utf-8")
        logging.info(f"tracked_changes.json saved to {output_json_path}")
    except Exception as e:
        logging.error(f"Failed to write tracked_changes.json: {e}", exc_info=True)


export_tracked_changes_to_json(DATABASE_FILE, OUTPUT_FOLDER)
