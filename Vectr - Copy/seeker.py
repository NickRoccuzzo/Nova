# seeker.py

# ---------------- MODULES ---------------- X
import os
import shutil
import time
import logging
import math
import json
import zipfile
import threading
import numpy as np
import pandas as pd
import yfinance as yf
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
from VectrPyLogic import save_options_data, preprocess_dates, find_top_volume_contracts, local_tz


# ---------------- CONFIG & DEFAULTS ----------------

SEEKER_DIR   = os.path.join(os.getcwd(), "seeker")
ARCHIVE_DIR  = os.path.join(os.getcwd(), "seeker_archive")
DB_PATH = os.path.join(os.getcwd(), "seeker_database.json")

DEFAULT_CONFIG = {
    "max_workers": 8,
    "max_retries": 3,
    "backoff_factor": 2,
    "retention_days": 1,
    "sector_pause": 20,          # <– new: seconds between sectors
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

def load_ticker_sector_industry(json_path="tickers.json") -> dict[str, dict[str,str]]:
    """
    Read tickers.json and return a dict:
      ticker -> {"sector": SectorName, "industry": IndustryName}
    """
    mapping: dict[str, dict[str,str]] = {}
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"tickers.json not found at {json_path}")
        return mapping

    for sector, industries in data.items():
        for industry, tickers in industries.items():
            for t in tickers:
                mapping[t] = {"sector": sector, "industry": industry}
    return mapping


def build_seeker_database(
    search_root: str = None,
    output_path: str = None,
    tickers_json: str = "tickers.json"
) -> dict[str, dict]:
    """
    Walk seeker/<TICKER>/top_volume_contracts.json for every ticker,
    consolidate into seeker_database.json, and inject sector/industry.
    """
    base = os.getcwd()
    root = search_root or os.path.join(base, "seeker")
    out_file = output_path or os.path.join(base, "seeker_database.json")

    # load our reverse‑lookup map
    ticker_map = load_ticker_sector_industry(tickers_json)

    db: dict[str, dict] = {}
    if not os.path.isdir(root):
        logging.warning(f"No seeker folder found at {root}, skipping DB build")
    else:
        for ticker in os.listdir(root):
            jf = os.path.join(root, ticker, "top_volume_contracts.json")
            if not os.path.isfile(jf):
                continue

            try:
                with open(jf, "r") as f:
                    entry = json.load(f)
            except Exception as e:
                logging.error(f"Failed loading JSON for {ticker}: {e}")
                continue

            # inject sector/industry if we have it
            meta = ticker_map.get(ticker, {})
            entry["sector"] = meta.get("sector")
            entry["industry"] = meta.get("industry")

            # Inject last_price (only once while connected)
            try:
                yf_ticker = yf.Ticker(ticker)
                price = yf_ticker.info.get("regularMarketPrice", 0.0)
                entry["last_price"] = float(price) if price else 0.0
            except Exception as e:
                logging.warning(f"Failed to fetch price for {ticker}: {e}")
                entry["last_price"] = 0.0

            db[ticker] = entry

    # write consolidated DB
    try:
        with open(out_file, "w") as f:
            json.dump(db, f, indent=4)
        logging.info(f"Wrote seeker database with {len(db)} tickers to {out_file}")
    except Exception as e:
        logging.error(f"Failed writing seeker database: {e}")

    return db

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
    Return, for each expiry, up to three highest‑volume CALLS and PUTS.
    """
    base  = os.getcwd()
    calls = preprocess_dates(os.path.join(base, ticker, "CALLS"), "CALLS")
    puts  = preprocess_dates(os.path.join(base, ticker, "PUTS"),  "PUTS")

    result = {}
    for side, data in (("CALLS", calls), ("PUTS", puts)):
        for exp, df in data.items():
            if df.empty:
                continue

            # numeric coercion (in case preprocess didn't already)
            df["volume"]       = pd.to_numeric(df["volume"],       errors="coerce")
            df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")

            # drop rows with no volume or OI
            df = df.dropna(subset=["volume", "openInterest"])
            if df.empty:
                continue

            top3 = df.nlargest(3, "volume")

            contracts = []
            for _, row in top3.iterrows():
                spent = row["volume"] * row["lastPrice"] * 100
                contracts.append({
                    "strike":        row["strike"],
                    "volume":        int(row["volume"]),
                    "openInterest":  int(row["openInterest"]),
                    "total_spent":   format_dollar_amount(spent),
                    "unusual":       row["volume"] > row["openInterest"],
                    "freshness": (
                        "current" if row["lastTradeDate_Local"].date() ==
                        datetime.now(local_tz).date() else "old"
                    ),
                    "lastTradeDate": format_trade_date(row["lastTradeDate_Local"]),
                })

            result.setdefault(exp, {"CALLS": [], "PUTS": []})[side] = contracts

    return result


def get_top_three_oi_contracts(ticker: str) -> dict:
    """
    For each expiry, return up to three contracts *per side*
    having the largest openInterest.  Structure mirrors
    get_top_three_contracts so down‑stream code can treat them alike.

        {
          "05/16/25": {
              "CALLS": [
                  {"strike": 150, "openInterest": 65432},
                  ...
              ],
              "PUTS": [ ... ]
          },
          ...
        }
    """
    base  = os.getcwd()
    calls = preprocess_dates(os.path.join(base, ticker, "CALLS"), "CALLS")
    puts  = preprocess_dates(os.path.join(base, ticker, "PUTS"),  "PUTS")

    result: dict[str, dict[str, list[dict]]] = {}

    for side, data in (("CALLS", calls), ("PUTS", puts)):
        for exp, df in data.items():
            if df.empty:
                continue

            df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
            df = df.dropna(subset=["openInterest"])
            if df.empty:
                continue

            top3 = df.nlargest(3, "openInterest")

            simplified = [
                {
                    "strike":       row["strike"],
                    "openInterest": int(row["openInterest"]),
                }
                for _, row in top3.iterrows()
            ]

            result.setdefault(exp, {"CALLS": [], "PUTS": []})[side] = simplified

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
        top_oi_contracts = get_top_three_oi_contracts(ticker)
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
            "by_volume": convert_np_types(top_volume_contracts),
            "by_oi": top_oi_contracts  # already plain ints
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



def _compute_setups(
    db: dict[str, dict],
    direction: str,      # "ABOVE" for bullish, "BELOW" for bearish
    top_n: int = 10
) -> list[dict]:
    """
    1) Compute raw, outlier‑boosted OI scores per ticker (Part 3 logic).
    2) Group tickers by sector and convert raw scores → 0–1 percentile within each sector.
    3) Return the top_n tickers sorted by descending percentile (then raw score).
    """
    # --- 1) Raw, boosted scores ---
    temp: list[dict] = []
    for ticker, ent in db.items():
        price = ent.get("last_price", 0.0)

        # collect per‑expiry OI sums
        oi_sums = []
        for sides in ent.get("by_oi", {}).values():
            calls, puts = sides.get("CALLS", []), sides.get("PUTS", [])
            if calls and puts:
                oi_sums.append(calls[0]["openInterest"] + puts[0]["openInterest"])

        if not oi_sums:
            raw_score = 0.0
        else:
            mean_oi = sum(oi_sums) / len(oi_sums)
            var     = sum((x - mean_oi) ** 2 for x in oi_sums) / len(oi_sums)
            std_oi  = math.sqrt(var) if var > 0 else 1.0

            raw_score = 0.0
            for sides in ent.get("by_oi", {}).values():
                calls, puts = sides.get("CALLS", []), sides.get("PUTS", [])
                if not (calls and puts):
                    continue
                call, put = calls[0], puts[0]
                oi_sum     = call["openInterest"] + put["openInterest"]
                max_call   = call["strike"]
                max_put    = put["strike"]

                hit = (
                    (direction == "ABOVE" and max_call > price and max_put < price)
                    or
                    (direction == "BELOW" and max_call < price and max_put > price)
                )
                if not hit:
                    continue

                z     = (oi_sum - mean_oi) / std_oi
                boost = 1 + z
                raw_score += oi_sum * boost

        temp.append({
            "ticker":    ticker,
            "sector":    ent.get("sector", "Unknown"),
            "raw_score": raw_score,
        })

    # --- 2) Percentile‑normalize within each sector ---
    sectors = defaultdict(list)
    for rec in temp:
        sectors[rec["sector"]].append(rec)

    for recs in sectors.values():
        recs.sort(key=lambda r: r["raw_score"])
        n = len(recs)
        for idx, r in enumerate(recs):
            # if only one ticker in sector, give it 1.0
            r["percentile"] = 1.0 if n == 1 else idx / (n - 1)

    # --- 3) Flatten & sort globally by percentile → raw_score ---
    all_recs = [r for recs in sectors.values() for r in recs]
    all_recs.sort(
        key=lambda r: (r["percentile"], r["raw_score"]),
        reverse=True
    )

    # Return top_n as [{"ticker":…, "score":percentile}, …]
    top = all_recs[:top_n]
    return [{"ticker": r["ticker"], "score": r["percentile"]} for r in top]


def _compute_sectors(db: dict[str, dict]) -> list[dict]:
    from math import isnan
    from statistics import pstdev

    # build sector → [tickers]
    sector_map: dict[str, list[str]] = {}
    for t, meta in db.items():
        sec = meta.get("sector") or "Unknown"
        sector_map.setdefault(sec, []).append(t)

    sector_scores: dict[str, float] = {}
    for sec, tickers in sector_map.items():
        stds = []
        for t in tickers:
            day_map = db[t].get("by_volume", {})
            vols = []
            for day in day_map.values():
                for side_key in ("CALLS", "PUTS"):
                    for c in day.get(side_key, []):
                        if c.get("unusual"):
                            v = c.get("volume", 0.0)
                            if v is not None and not isnan(v):
                                vols.append(v)
            if vols:
                stds.append(pstdev(vols))
        sector_scores[sec] = (sum(stds) / len(stds)) if stds else 0.0

    sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"sector": sec, "score": score} for sec, score in sorted_sectors]

if __name__ == "__main__":
    # Archive old data and prune old archives before starting
    archive_and_clear_seeker()
    prune_old_archives()

    # Load tickers by sector
    sectors_dict = get_tickers_by_sector("tickers.json")
    for sector, tickers in sectors_dict.items():
        logging.info(f"Sector '{sector}' has {len(tickers)} tickers.")

    # Hard-coded 10-minute interval
    interval_seconds = 10 * 60
    logging.info("Scheduled query interval set to 10 minutes.")

    # Start listener for manual updates ("u" + Enter)
    listener_thread = threading.Thread(target=input_listener, daemon=True)
    listener_thread.start()

    last_run_time = time.time()
    while True:
        now = time.time()
        if (now - last_run_time) >= interval_seconds or manual_update_event.is_set():
            logging.info("Starting update...")
            for sector, tickers_list in sectors_dict.items():
                logging.info(f"Processing sector: {sector} with {len(tickers_list)} tickers...")
                process_tickers(tickers_list)
                build_seeker_database()  # ← writes top‑level tickers

                # ─── inject precomputed setups & sectors ───
                DB_PATH = os.path.join(os.getcwd(), "seeker_database.json")
                with open(DB_PATH, "r") as f:
                    full = json.load(f)

                db = full  # flat JSON: each key is a ticker
                setups_bullish = _compute_setups(db, "ABOVE", top_n=20)
                setups_bearish = _compute_setups(db, "BELOW", top_n=20)
                full["setups"] = {
                    "bullish": setups_bullish,
                    "bearish": setups_bearish
                }
                full["sectors"] = _compute_sectors(db)

                with open(DB_PATH, "w") as f:
                    json.dump(full, f, indent=2)
                # ────────────────────────────────────────────

                # Pause 5 minutes between sectors to respect API limits (! Very important, this is the sweet spot for timing ! )
                time.sleep(CONFIG["sector_pause"])
            last_run_time = time.time()
            manual_update_event.clear()
            logging.info("Concurrent update completed for all sectors. Waiting for next scheduled update...")
        time.sleep(5)





