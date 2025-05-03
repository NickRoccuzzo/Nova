# seeker.py

# ---------------- MODULES ---------------- X

import os
import shutil
import time
import logging
import json
import zipfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler

from VectrPyLogic import save_options_data


# ---------------- CONFIG & DEFAULTS ----------------
DEFAULT_CONFIG = {
    "max_workers": 2,
    "max_retries": 3,
    "backoff_factor": 2,
    "retention_days": 1  # prune archives older than this many days
}

if os.path.exists("config.json"):
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = DEFAULT_CONFIG


# ----------------- LOGGING SETUP ------------------
log_file = "seeker.log"
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Rotate when log > 5 MB, keep 5 backups
rot_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
rot_handler.setFormatter(log_formatter)
rot_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[rot_handler, console_handler])

# Global event used to trigger a manual update.
manual_update_event = threading.Event()


# --------- ARCHIVE OLD SEEKER + PRUNE -------------

def prune_old_archives():
    """Delete zip archives in seeker_archive older than retention_days."""
    archive_dir = os.path.join(os.getcwd(), "seeker_archive")
    if not os.path.isdir(archive_dir):
        return

    cutoff = datetime.now() - timedelta(days=CONFIG["retention_days"])
    for fname in os.listdir(archive_dir):
        if fname.lower().endswith(".zip"):
            path = os.path.join(archive_dir, fname)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if mtime < cutoff:
                try:
                    os.remove(path)
                    logging.info(f"Pruned old archive: {fname}")
                except Exception as e:
                    logging.error(f"Failed pruning {fname}: {e}")

def archive_and_clear_seeker():
    """If seeker/ exists, zip it into seeker_archive and then remove it."""
    base = os.getcwd()
    seeker_dir = os.path.join(base, "seeker")
    if not os.path.isdir(seeker_dir):
        return

    # ensure archive folder
    archive_root = os.path.join(base, "seeker_archive")
    os.makedirs(archive_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = os.path.join(archive_root, f"seeker_{timestamp}.zip")

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(seeker_dir):
            for f in files:
                full = os.path.join(root, f)
                rel  = os.path.relpath(full, seeker_dir)
                zf.write(full, rel)
    logging.info(f"Archived old seeker data to {zip_name}")

    # remove the old seeker folder
    shutil.rmtree(seeker_dir)
    logging.info("Removed old seeker/ folder for a clean slate")


# ---------------- Helper Functions ---------------- X
def convert_np_types(data):
    """Recursively convert NumPy data types to native Python types."""
    if isinstance(data, dict):
        return {k: convert_np_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_np_types(item) for item in data]
    elif isinstance(data, np.generic):
        return data.item()
    else:
        return data


def format_dollar_amount(amount):
    """Format a numeric dollar amount into a human-readable string."""
    if amount >= 1e6:
        return f"${amount / 1e6:.1f}M"
    elif amount >= 1e3:
        return f"${amount / 1e3:.1f}K"
    else:
        return f"${amount:.2f}"


def save_search_data(ticker: str) -> None:
    """
    Copy the CALLS and PUTS folders for a given ticker to the central 'seeker/config1/searches'
    folder. If the target folders already exist, they are removed first.
    """
    base_dir = os.getcwd()
    source_folder = os.path.join(base_dir, ticker)
    target_folder = os.path.join(base_dir, "seeker", ticker)
    os.makedirs(target_folder, exist_ok=True)
    for option_type in ["CALLS", "PUTS"]:
        source = os.path.join(source_folder, option_type)
        dest = os.path.join(target_folder, option_type)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        try:
            shutil.copytree(source, dest)
            logging.info(f"Copied {option_type} data for {ticker} to {dest}")
        except Exception as e:
            logging.error(f"Error copying {option_type} data for {ticker}: {e}")


def format_trade_date(dt):
    """
    Given a timezone-aware datetime object, return a human-readable string in the format:
    "4/4/2025 3:46:46 PM EDT"
    """
    # Format with leading zeros.
    formatted = dt.strftime("%m/%d/%Y %I:%M:%S %p EDT")
    # Remove leading zeros from month and day.
    date_part, time_part, meridiem, tz = formatted.split(" ")
    month, day, year = date_part.split("/")
    month = month.lstrip("0")
    day = day.lstrip("0")
    return f"{month}/{day}/{year} {time_part} {meridiem} {tz}"


def get_top_three_contracts(ticker: str) -> dict:
    """
    Reads CSV files from the ticker's CALLS and PUTS folders, extracts the top three contracts (by volume)
    for each expiration date, and returns a dictionary mapping expiration dates to their corresponding
    CALLS and PUTS data.
    [Documentation omitted for brevity]
    """
    result = {}
    base_dir = os.getcwd()
    ticker_folder = os.path.join(base_dir, ticker)
    today_date = datetime.now(ZoneInfo("America/New_York")).date()

    for option_type in ["CALLS", "PUTS"]:
        option_folder = os.path.join(ticker_folder, option_type)
        if not os.path.exists(option_folder):
            continue
        csv_files = [f for f in os.listdir(option_folder) if f.endswith(".csv")]
        for csv_file in csv_files:
            try:
                exp_date_raw = csv_file[:8]
                year = exp_date_raw[0:4]
                month = exp_date_raw[4:6]
                day = exp_date_raw[6:8]
                exp_date = f"{month}/{day}/{year}"
            except Exception as e:
                logging.error(f"Error parsing expiration date from filename {csv_file}: {e}")
                continue

            file_path = os.path.join(option_folder, csv_file)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logging.error(f"Error reading CSV file {file_path}: {e}")
                continue

            if 'volume' not in df.columns or 'lastTradeDate' not in df.columns:
                logging.error(f"Required columns not found in {file_path}")
                continue

            try:
                df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'], errors='coerce')
                if df['lastTradeDate'].dt.tz is None:
                    df['lastTradeDate'] = df['lastTradeDate'].dt.tz_localize('UTC')
                df['lastTradeDate_EST'] = df['lastTradeDate'].dt.tz_convert(ZoneInfo("America/New_York"))
                df['is_current'] = df['lastTradeDate_EST'].dt.date.apply(lambda d: 1 if d == today_date else 0)
            except Exception as e:
                logging.error(f"Error processing 'lastTradeDate' in {file_path}: {e}")
                continue

            df = df.sort_values(by=['is_current', 'volume'], ascending=[False, False])
            top_three = df.head(3)

            contracts = []
            for _, row in top_three.iterrows():
                try:
                    strike = row['strike']
                    volume = row['volume']
                    openInterest = row['openInterest']
                    lastPrice = row['lastPrice']
                    total_spent = volume * lastPrice * 100
                    formatted_total_spent = format_dollar_amount(total_spent)
                    unusual = volume > openInterest
                    freshness = "current" if row['is_current'] == 1 else "old"
                    # Format the lastTradeDate_EST into a human-readable string.
                    last_trade_est = row['lastTradeDate_EST']
                    formatted_trade_date = format_trade_date(last_trade_est)
                    contract_data = {
                        "strike": strike,
                        "volume": volume,
                        "openInterest": openInterest,
                        "total_spent": formatted_total_spent,
                        "unusual": unusual,
                        "freshness": freshness,
                        "lastTradeDate": formatted_trade_date
                    }
                    contracts.append(contract_data)
                except Exception as e:
                    logging.error(f"Error processing row in {file_path}: {e}")
            if exp_date not in result:
                result[exp_date] = {"CALLS": [], "PUTS": []}
            result[exp_date][option_type] = contracts

    return result


def safe_save_options_data(ticker, max_retries=3, backoff_factor=2):
    """
    Wraps the save_options_data call with retry logic.
    Tries up to max_retries times with exponential backoff.
    """
    retries = 0
    while retries < max_retries:
        try:
            save_options_data(ticker)
            return True
        except Exception as e:
            retries += 1
            wait = backoff_factor ** retries
            logging.error(f"Error in save_options_data for {ticker}: {e}. Retrying in {wait} seconds (Attempt {retries}/{max_retries})")
            time.sleep(wait)
    logging.error(f"Failed to save options data for {ticker} after {max_retries} attempts.")
    return False


def process_ticker(ticker: str) -> None:
    """
    Process a single ticker:
      1. Fetch and save the options data (CSV files) using safe_save_options_data.
      2. Extract the top three volume contracts per expiration date.
      3. If no option chain is found, retry after 10 seconds.
         If it still fails, log the error, clean up any temporary folders, and skip this ticker.
      4. Archive any existing JSON, then save the new data.
      5. Copy the folders to the central searches folder.
      6. Clean up the temporary ticker folder.
    """
    logging.info(f"Processing ticker: {ticker}")
    try:
        # Step 1: Fetch and save option chain CSVs using safe API call.
        if not safe_save_options_data(ticker, max_retries=CONFIG["max_retries"], backoff_factor=CONFIG["backoff_factor"]):
            logging.error(f"Failed to fetch data for {ticker} after retries; skipping ticker.")
            return

        # Step 2: Extract top three contracts data.
        top_volume_contracts = get_top_three_contracts(ticker)
        if not top_volume_contracts or top_volume_contracts == {}:
            logging.error(f"No option chain found for ticker {ticker}, may not exist. Retrying in 10 seconds...")
            time.sleep(10)
            top_volume_contracts = get_top_three_contracts(ticker)
            if not top_volume_contracts or top_volume_contracts == {}:
                logging.error(f"Ticker {ticker} still failed after retry; skipping this ticker.")
                # Remove any temporary folder that might have been created.
                temp_folder = os.path.join(os.getcwd(), ticker)
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
                return

        top_volume_contracts_converted = convert_np_types(top_volume_contracts)

        # Prepare output data with a timestamp.
        output_data = {
            "timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "data": top_volume_contracts_converted
        }

        # Step 3: Create the target folder and save output_data.
        base_dir = os.getcwd()
        target_folder = os.path.join(base_dir, "seeker", ticker)
        os.makedirs(target_folder, exist_ok=True)
        json_file_path = os.path.join(target_folder, "top_volume_contracts.json")

        # Archive previous JSON if it exists.
        if os.path.exists(json_file_path):
            archive_folder = os.path.join(target_folder, "archive")
            os.makedirs(archive_folder, exist_ok=True)
            timestamp_for_file = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")
            archive_file_path = os.path.join(archive_folder, f"top_volume_contracts_{timestamp_for_file}.json")
            shutil.copy2(json_file_path, archive_file_path)
            logging.info(f"Archived previous JSON to {archive_file_path}")

        with open(json_file_path, "w") as f:
            json.dump(output_data, f, indent=4)
        logging.info(f"Saved top_volume_contracts data for {ticker} to {json_file_path}")

        # Step 4: Copy the downloaded data to the central searches folder.
        save_search_data(ticker)

        # Step 5: Clean up the temporary ticker folder.
        temp_folder = os.path.join(os.getcwd(), ticker)
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

        logging.info(f"Ticker {ticker} processed successfully.")
    except Exception as e:
        logging.error(f"Error processing ticker {ticker}: {e}")


def build_seeker_database(search_root=None, output_path=None):
    """
    Walks through seeker/<TICKER>/top_volume_contracts.json for every ticker
    and writes a single JSON mapping ticker -> its data.

    :param search_root: folder where all ticker sub-folders live (default: ./seeker)
    :param output_path:   path to write the DB JSON (default: ./seeker_database.json)
    :return: dict of the consolidated data
    """
    base = os.getcwd()
    root = search_root or os.path.join(base, "seeker")
    out_file = output_path or os.path.join(base, "seeker_database.json")

    db = {}
    if not os.path.isdir(root):
        logging.warning(f"No seeker folder found at {root}, skipping DB build")
    else:
        for ticker in os.listdir(root):
            tdir = os.path.join(root, ticker)
            jf = os.path.join(tdir, "top_volume_contracts.json")
            if os.path.isfile(jf):
                try:
                    with open(jf, "r") as f:
                        db[ticker] = json.load(f)
                except Exception as e:
                    logging.error(f"Failed loading JSON for {ticker}: {e}")

    # write consolidated DB
    try:
        with open(out_file, "w") as f:
            json.dump(db, f, indent=4)
        logging.info(f"Wrote seeker database with {len(db)} tickers to {out_file}")
    except Exception as e:
        logging.error(f"Failed writing seeker database: {e}")

    return db


def get_tickers_by_sector(json_path="tickers.json"):
    """Load tickers from a JSON file and return a dict with sector names as keys and a flat list of tickers as values."""
    with open(json_path, "r") as f:
        data = json.load(f)
    sectors = {}
    for sector, industries in data.items():
        tickers = []
        for industry, ticker_list in industries.items():
            tickers.extend(ticker_list)
        sectors[sector] = tickers
    return sectors

def process_tickers(tickers_list) -> None:
    """
    Process a list of tickers concurrently using ThreadPoolExecutor.
    The number of concurrent workers is controlled by max_workers.
    """
    max_workers = CONFIG["max_workers"]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(process_ticker, ticker): ticker for ticker in tickers_list}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                future.result()
                logging.info(f"Ticker {ticker} processed successfully (concurrently).")
            except Exception as exc:
                logging.error(f"Ticker {ticker} generated an exception: {exc}")


def input_listener():
    """
    Listen for user input to trigger a manual update.
    Type 'u' and press Enter to trigger an immediate update.
    """
    while True:
        cmd = input("Type 'u' and press Enter for an immediate update (or wait for the scheduled update): ")
        if cmd.strip().lower() == 'u':
            manual_update_event.set()


if __name__ == "__main__":
    # Archive old data and prune old archives before starting
    archive_and_clear_seeker()
    prune_old_archives()

    # Load tickers by sector
    sectors_dict = get_tickers_by_sector("tickers.json")
    for sector, tickers in sectors_dict.items():
        logging.info(f"Sector '{sector}' has {len(tickers)} tickers.")

    # Hard-coded 15-minute interval
    interval_seconds = 15 * 60
    logging.info("Scheduled query interval set to 15 minutes.")

    # Start listener for manual updates ("u" + Enter)
    listener_thread = threading.Thread(target=input_listener, daemon=True)
    listener_thread.start()

    last_run_time = time.time()
    while True:
        now = time.time()
        if (now - last_run_time) >= interval_seconds or manual_update_event.is_set():
            logging.info("Starting update of tickers concurrently, sector by sector...")
            for sector, tickers_list in sectors_dict.items():
                logging.info(f"Processing sector: {sector} with {len(tickers_list)} tickers...")
                process_tickers(tickers_list)
                build_seeker_database()
                # Pause 5 minutes between sectors to respect API limits
                time.sleep(300)
            last_run_time = time.time()
            manual_update_event.clear()
            logging.info("Concurrent update completed for all sectors. Waiting for next scheduled update...")
        time.sleep(5)