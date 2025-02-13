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
    filename="novapylogic.log",
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

    calls_folder = os.path.join(ticker_path, "CALLS")
    puts_folder = os.path.join(ticker_path, "PUTS")

    if not os.path.exists(calls_folder) or not os.path.exists(puts_folder):
        logging.warning(f"Missing data for {ticker}. Skipping...")
        return {}

    calls_data = preprocess_dates(calls_folder, "CALLS")
    puts_data = preprocess_dates(puts_folder, "PUTS")

    calls_oi = {date: df['openInterest'].sum() for date, df in calls_data.items() if not df.empty}
    puts_oi = {date: df['openInterest'].sum() for date, df in puts_data.items() if not df.empty}
    calls_volume = {date: df['volume'].sum() for date, df in calls_data.items() if not df.empty}
    puts_volume = {date: df['volume'].sum() for date, df in puts_data.items() if not df.empty}

    logging.info(f"Processed OI -> Calls: {calls_oi}, Puts: {puts_oi}")
    logging.info(f"Processed Volume -> Calls: {calls_volume}, Puts: {puts_volume}")

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

    def process_option_data(option_data, max_strike_dicts, option_type):
        for date, df in option_data.items():
            if not df.empty:
                sorted_data = df.sort_values(by='openInterest', ascending=False)

                max_strike_dicts[0][date] = sorted_data.iloc[0]['strike'] if len(sorted_data) > 0 else 0
                max_strike_dicts[1][date] = sorted_data.iloc[1]['strike'] if len(sorted_data) > 1 else 0
                max_strike_dicts[2][date] = sorted_data.iloc[2]['strike'] if len(sorted_data) > 2 else 0

                # ✅ Fix idxmax() bug: Check if DataFrame is empty before calling idxmax()
                if not df.empty and df['volume'].notna().any():
                    highest_volume_idx = df['volume'].idxmax()
                    highest_volume = df.loc[highest_volume_idx]
                    total_spent = highest_volume['volume'] * highest_volume['lastPrice'] * 100
                    formatted_spent = format_dollar_amount(total_spent)

                    unusual = highest_volume['volume'] > highest_volume['openInterest']

                    top_volume_contracts.append({
                        'type': option_type,
                        'strike': highest_volume['strike'],
                        'volume': highest_volume['volume'],
                        'openInterest': highest_volume['openInterest'],
                        'date': date,
                        'total_spent': formatted_spent,
                        'unusual': unusual
                    })

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

    return {
        "calls_oi": calls_oi,
        "puts_oi": puts_oi,
        "calls_volume": calls_volume,
        "puts_volume": puts_volume,
        "max_strike_calls": max_strike_calls,
        "second_max_strike_calls": second_max_strike_calls,
        "third_max_strike_calls": third_max_strike_calls,
        "max_strike_puts": max_strike_puts,
        "second_max_strike_puts": second_max_strike_puts,
        "third_max_strike_puts": third_max_strike_puts,
        "avg_strike": avg_strike,
        "top_volume_contracts": top_volume_contracts,
        "current_price": current_price,
        "company_name": company_name
    }
