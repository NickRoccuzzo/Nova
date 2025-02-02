# -- MODULES -- #
import os
import pandas as pd
import time
import numpy as np
import json
import logging
from datetime import datetime
import yfinance as yf

# Configure logging
logging.basicConfig(
    filename="nova_logic.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Use Docker volume path for shared storage
TICKER_DIR = os.getenv("TICKER_DIR", "/shared_data")


def preprocess_dates(data_dir, file_suffix):
    """
    Preprocess and sort option chain data by expiration dates.
    """
    sorted_data = {}

    if not os.path.exists(data_dir):
        logging.warning(f"Directory {data_dir} does not exist. Skipping...")
        return sorted_data

    for filename in os.listdir(data_dir):
        if filename.endswith(file_suffix + ".csv"):
            try:
                date_str = filename.replace("_", "").split(file_suffix)[0]
                expiration_date = datetime.strptime(date_str, '%Y%m%d')
                formatted_date = expiration_date.strftime('%m/%d/%y')

                file_path = os.path.join(data_dir, filename)
                df = pd.read_csv(file_path)

                if "openInterest" not in df.columns or "strike" not in df.columns:
                    logging.warning(f"Missing columns in {file_path}. Available: {df.columns}")

                sorted_data[formatted_date] = df

            except ValueError as e:
                logging.error(f"Error processing {filename}: {e}")

    return dict(sorted(sorted_data.items(), key=lambda x: datetime.strptime(x[0], '%m/%d/%y')))


def fetch_stock_data(ticker):
    """
    Fetch stock price data with retries.
    """
    stock = yf.Ticker(ticker)

    for attempt in range(5):  
        try:
            current_data = stock.history(period="1d")
            if not current_data.empty:
                return current_data['Close'].iloc[-1]
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            time.sleep(2 ** attempt)  

    return 0.0   # Default value if all attempts fail


def format_dollar_amount(amount):
    """
    Format numbers into human-readable currency format.
    """
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.1f}K"
    else:
        return f"${amount:.2f}"


def gather_options_data(ticker):
    """
    Extracts insights from options chain data.
    """
    ticker_path = os.path.join(TICKER_DIR, ticker)

    calls_dir = os.path.join(ticker_path, "CALLS")
    puts_dir = os.path.join(ticker_path, "PUTS")

    if not os.path.exists(calls_dir) or not os.path.exists(puts_dir):
        logging.warning(f"Missing data for {ticker}. Skipping...")
        return {}

    calls_data = preprocess_dates(calls_dir, "CALLS")
    puts_data = preprocess_dates(puts_dir, "PUTS")

    calls_oi = {date: df['openInterest'].sum() for date, df in calls_data.items() if not df.empty}
    puts_oi = {date: df['openInterest'].sum() for date, df in puts_data.items() if not df.empty}

    logging.info(f"Processed OI -> Calls: {calls_oi}, Puts: {puts_oi}")

    max_strike_calls, second_max_strike_calls, third_max_strike_calls = {}, {}, {}
    max_strike_puts, second_max_strike_puts, third_max_strike_puts = {}, {}, {}
    avg_strike = {}
    top_volume_contracts = []

    try:
        stock = yf.Ticker(ticker)
        current_price = fetch_stock_data(ticker)
        company_name = stock.info.get('longName', 'N/A')
    except Exception as e:
        logging.error(f"Failed to fetch data for {ticker}: {e}")
        current_price, company_name = 0.0, "Unknown"

    process_option_data(calls_data, [max_strike_calls, second_max_strike_calls, third_max_strike_calls], "CALL")
    process_option_data(puts_data, [max_strike_puts, second_max_strike_puts, third_max_strike_puts], "PUT")

    for date in max_strike_calls.keys():
        if date in max_strike_puts and (calls_oi.get(date, 0) + puts_oi.get(date, 0) > 0):
            total_oi = calls_oi.get(date, 0) + puts_oi.get(date, 0)
            if total_oi > 0:
                avg_strike[date] = (
                    (max_strike_calls[date] * calls_oi.get(date, 0) +
                     max_strike_puts[date] * puts_oi.get(date, 0)) / total_oi
                )
            else:
                avg_strike[date] = np.nan

    # ✅ Get sector from the container's environment variable
    sector = os.getenv("SECTOR", "Unknown")

    # ✅ Find the industry dynamically from `tickers.json` for the given sector
    industry = "Unknown"
    tickers_mapping_file = os.path.join(TICKER_DIR, "tickers.json")

    try:
        with open(tickers_mapping_file, "r") as f:
            tickers_mapping = json.load(f)

        # ✅ Get the industries for the current sector
        sector_data = tickers_mapping.get(sector, {})

        # ✅ Loop through industries to find where the ticker exists
        for industry_name, tickers_list in sector_data.items():
            if ticker in tickers_list:
                industry = industry_name
                break  # ✅ Stop searching once we find the industry

    except Exception as e:
        logging.error(f"Error reading {tickers_mapping_file}: {e}")

    # ✅ Update the final result dictionary
    return {
        "calls_oi": calls_oi,
        "puts_oi": puts_oi,
        "max_strike_calls": max_strike_calls,
        "second_max_strike_calls": second_max_strike_calls,
        "third_max_strike_calls": third_max_strike_calls,
        "max_strike_puts": max_strike_puts,
        "second_max_strike_puts": second_max_strike_puts,
        "third_max_strike_puts": third_max_strike_puts,
        "avg_strike": avg_strike,
        "top_volume_contracts": top_volume_contracts,
        "current_price": current_price,
        "company_name": company_name,
        "sector": sector,   # ✅ Now directly included
        "industry": industry  # ✅ Dynamically assigned
    }

