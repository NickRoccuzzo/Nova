# -- MODULES -- #
import os
import pandas as pd
import time
import numpy as np
import json
import logging
from datetime import datetime
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    filename="nova_logic.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Use Docker volume path for shared storage
TICKER_DIR = os.getenv("TICKER_DIR", "/shared_data")

# Create a session with retry logic to handle 429 errors
session = requests.Session()
retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))


def preprocess_dates(data_dir, file_suffix):
    """
    Preprocess and sort option chain data by expiration dates.

    Parameters:
    - data_dir (str): Directory containing options data CSVs.
    - file_suffix (str): 'CALLS' or 'PUTS' for filtering files.

    Returns:
    - dict: A dictionary with formatted dates as keys and DataFrames as values.
    """
    sorted_data = {}

    if not os.path.exists(data_dir):
        logging.warning(f"Directory {data_dir} does not exist. Skipping...")
        return sorted_data

    for filename in os.listdir(data_dir):
        if filename.endswith(file_suffix + ".csv"):
            try:
                # ✅ Fix: Remove underscores before extracting date
                date_str = filename.replace("_", "").split(file_suffix)[0]

                expiration_date = datetime.strptime(date_str, '%Y%m%d')
                formatted_date = expiration_date.strftime('%m/%d/%y')

                file_path = os.path.join(data_dir, filename)
                df = pd.read_csv(file_path)

                # ✅ Handle missing columns: Log a warning instead of skipping
                if "openInterest" not in df.columns or "strike" not in df.columns:
                    logging.warning(f"Missing required columns in {file_path}. Available columns: {df.columns}")

                sorted_data[formatted_date] = df  # Store DataFrame even if columns are missing

            except ValueError as e:
                logging.error(f"Error processing {filename}: {e}")

    return dict(sorted(sorted_data.items(), key=lambda x: datetime.strptime(x[0], '%m/%d/%y')))


def fetch_stock_data(ticker):
    """
    Fetch stock price data with retries and exponential backoff to handle rate limits.
    """
    stock = yf.Ticker(ticker)

    for attempt in range(5):  # Retry up to 5 times
        try:
            current_data = stock.history(period="1d")
            if not current_data.empty:
                return current_data['Close'].iloc[-1]
            else:
                return 0.0
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

    return 0.0  # Return 0 if all retries fail


def gather_options_data(ticker):
    """
    Extracts useful insights from options chain data.

    Returns:
    - Dictionary containing summarized options data.
    """
    ticker_path = os.path.join(TICKER_DIR, ticker)

    calls_dir = os.path.join(ticker_path, "CALLS")
    puts_dir = os.path.join(ticker_path, "PUTS")

    if not os.path.exists(calls_dir) or not os.path.exists(puts_dir):
        logging.warning(f"Missing data for {ticker}. Skipping...")
        return {}

    logging.info(f"Processing {ticker}: Calls Dir -> {calls_dir}, Puts Dir -> {puts_dir}")

    calls_data = preprocess_dates(calls_dir, "CALLS")
    puts_data = preprocess_dates(puts_dir, "PUTS")

    logging.info(f"Extracted {len(calls_data)} call expirations and {len(puts_data)} put expirations for {ticker}")

    calls_oi = {date: df['openInterest'].sum() for date, df in calls_data.items() if not df.empty}
    puts_oi = {date: df['openInterest'].sum() for date, df in puts_data.items() if not df.empty}

    logging.info(f"Processed Open Interest -> Calls: {calls_oi}, Puts: {puts_oi}")

    max_strike_calls, max_strike_puts = {}, {}

    # Fetch current stock price
    try:
        stock = yf.Ticker(ticker)
        current_price = fetch_stock_data(ticker)
        company_name = stock.info.get('longName', 'N/A')
    except Exception as e:
        logging.error(f"Failed to fetch data for {ticker}: {e}")
        current_price, company_name = 0.0, "Unknown"

    def process_option_data(option_data, max_strike_dict):
        for date, df in option_data.items():
            if not df.empty:
                sorted_data = df.sort_values(by='openInterest', ascending=False)
                max_strike_dict[date] = sorted_data.iloc[0]['strike'] if not sorted_data.empty else 0

    process_option_data(calls_data, max_strike_calls)
    process_option_data(puts_data, max_strike_puts)

    return {
        "calls_oi": calls_oi,
        "puts_oi": puts_oi,
        "max_strike_calls": max_strike_calls,
        "max_strike_puts": max_strike_puts,
        "current_price": current_price,
        "company_name": company_name
    }


def process_tickers_in_parallel(tickers):
    """
    Runs gather_options_data() in parallel for multiple tickers.

    Parameters:
    - tickers (list): List of tickers to process.
    """
    logging.info(f"Processing {len(tickers)} tickers in parallel...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(gather_options_data, tickers))

    logging.info("Parallel processing complete.")
    return results


if __name__ == "__main__":
    # Get assigned sector from environment variable (Docker will set this)
    SECTOR = os.getenv("SECTOR")

    if not SECTOR:
        raise ValueError("SECTOR environment variable is not set. Each container must be assigned a sector.")

    # Load tickers from tickers.json based on sector
    with open("tickers.json", "r") as f:
        tickers_data = json.load(f)

    if SECTOR not in tickers_data:
        raise ValueError(f"Sector '{SECTOR}' not found in tickers.json.")

    # Extract tickers for this sector
    tickers = [ticker for industry in tickers_data[SECTOR].values() for ticker in industry]

    # Run parallel processing
    process_tickers_in_parallel(tickers)
