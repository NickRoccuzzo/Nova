import os
import json
import time
import logging
import math
from pathlib import Path
from typing import Any

import yfinance as yf
import numpy as np

from NovaPyLogic import gather_options_data

# Configure logging
logging.basicConfig(
    filename="nova.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Get assigned sector from Docker ENV
SECTOR = os.getenv("SECTOR")
if not SECTOR:
    raise ValueError("SECTOR environment variable is not set. Each container must be assigned a sector.")

# Load tickers for the assigned sector
tickers_file = Path("tickers.json")
if not tickers_file.exists():
    raise FileNotFoundError("tickers.json not found.")

with tickers_file.open("r") as f:
    tickers_data = json.load(f)

if SECTOR not in tickers_data:
    raise ValueError(f"Sector '{SECTOR}' not found in tickers.json.")

industries = tickers_data[SECTOR]


def save_options_data(stock: yf.Ticker, ticker: str, exp_dates: list) -> None:
    """
    Retrieve options chain data for each expiration date and save as CSV.
    """
    base_folder = Path("/shared_data")
    ticker_folder = base_folder / ticker
    ticker_folder.mkdir(parents=True, exist_ok=True)

    calls_folder = ticker_folder / "CALLS"
    puts_folder = ticker_folder / "PUTS"
    calls_folder.mkdir(exist_ok=True)
    puts_folder.mkdir(exist_ok=True)

    if not exp_dates:
        logging.warning(f"No option chain found for {ticker}.")
        return

    for date in exp_dates:
        try:
            time.sleep(0.10)  # To reduce the risk of throttling
            opt = stock.option_chain(date)
            calls_csv = calls_folder / f"{date.replace('-', '')}_CALLS.csv"
            puts_csv = puts_folder / f"{date.replace('-', '')}_PUTS.csv"
            opt.calls.to_csv(calls_csv)
            opt.puts.to_csv(puts_csv)
        except Exception as e:
            logging.error(f"Error processing {ticker} for expiration {date}: {e}", exc_info=True)


def process_ticker(ticker: str) -> None:
    """
    Process a single ticker: download options data, generate JSON summary, and save files.
    """
    try:
        stock = yf.Ticker(ticker)
        exp_dates = stock.options

        # Save raw CSVs for options data
        save_options_data(stock, ticker, exp_dates)

        # Gather additional insights and write JSON output
        data_dict = gather_options_data(ticker, stock)
        clean_data = convert_keys_for_json(data_dict)

        output_path = Path("/shared_data") / ticker / f"{ticker}_raw.json"
        with output_path.open("w") as f_out:
            json.dump(clean_data, f_out, indent=2)

        logging.info(f"Successfully processed {ticker}")
    except Exception as e:
        logging.error(f"Failed processing {ticker}: {e}", exc_info=True)

    time.sleep(0.35)


def convert_keys_for_json(obj: Any) -> Any:
    """
    Recursively convert NumPy types into JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {str(k): convert_keys_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_for_json(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return round(float(obj), 2) if not math.isnan(obj) else None
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    """
    Main execution: process all tickers sequentially.
    """
    all_tickers = [ticker for industry in industries.values() for ticker in industry]
    logging.info(f"Processing {len(all_tickers)} tickers for sector '{SECTOR}' sequentially.")
    for ticker in all_tickers:
        process_ticker(ticker)
    logging.info(f"All tickers in sector '{SECTOR}' processed successfully.")


if __name__ == "__main__":
    main()
