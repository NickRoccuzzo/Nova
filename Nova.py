import os
import json
import yfinance as yf
import numpy as np
import math
import time
import logging
import concurrent.futures  # For parallel execution
from NovaPyLogic import gather_options_data


# Configure logging
logging.basicConfig(
    filename="nova.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Get assigned sector from Docker ENV (if a sector isn't assigned, error message will be written) #
SECTOR = os.getenv("SECTOR")

if not SECTOR:
    raise ValueError("SECTOR environment variable is not set. Each container must be assigned a sector.")

# Load tickers for the assigned sector
with open("tickers.json", "r") as f:
    tickers_data = json.load(f)

if SECTOR not in tickers_data:
    raise ValueError(f"Sector '{SECTOR}' not found in tickers.json.")

industries = tickers_data[SECTOR]  # Industries within this sector


def save_options_data(ticker):
    """ Retrieves options chain data for a given ticker and stores results as CSV. """
    base_folder = "/shared_data"  # Use shared Docker volume
    ticker_folder = os.path.join(base_folder, ticker)
    os.makedirs(ticker_folder, exist_ok=True)

    calls_folder = os.path.join(ticker_folder, "CALLS")
    puts_folder = os.path.join(ticker_folder, "PUTS")
    os.makedirs(calls_folder, exist_ok=True)
    os.makedirs(puts_folder, exist_ok=True)

    stock = yf.Ticker(ticker)
    exp_dates = stock.options

    if not exp_dates:
        logging.warning(f"No option chain found for {ticker}.")
        return

    for date in exp_dates:
        try:
            opt = stock.option_chain(date)
            opt.calls.to_csv(os.path.join(calls_folder, f"{date.replace('-', '')}_CALLS.csv"))
            opt.puts.to_csv(os.path.join(puts_folder, f"{date.replace('-', '')}_PUTS.csv"))
        except Exception as e:
            logging.error(f"Error processing {ticker} for expiration {date}: {e}")


def process_ticker(ticker):
    """Processes a single ticker: downloads CSVs, gathers data, and saves JSON output."""
    save_options_data(ticker)

    try:
        data_dict = gather_options_data(ticker)
        clean_data = convert_keys_for_json(data_dict)
        output_path = os.path.join("/shared_data", ticker, f"{ticker}_raw.json")
        with open(output_path, "w") as f_out:
            json.dump(clean_data, f_out, indent=2)
        logging.info(f"Successfully processed {ticker}")
    except Exception as e:
        logging.error(f"Failed processing {ticker}: {e}")

    # Add a delay to help avoid rate limiting.
    time.sleep(5)  # Adjust the delay as needed.


def convert_keys_for_json(obj):
    """ Converts NumPy values into JSON-friendly types. """
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


def main():
    """ Main execution: Fetches options data for all tickers in this sector in parallel. """
    all_tickers = [ticker for industry in industries.values() for ticker in industry]

    logging.info(f"Processing {len(all_tickers)} tickers for sector '{SECTOR}'.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(process_ticker, all_tickers)

    logging.info(f"All tickers in sector '{SECTOR}' processed successfully.")


if __name__ == "__main__":
    main()
