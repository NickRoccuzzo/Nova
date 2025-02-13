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
                # e.g. 20230217_CALLS.csv -> date_str = '20230217'
                date_str = filename.replace("_", "").split(file_suffix)[0]
                expiration_date = datetime.strptime(date_str, '%Y%m%d')
                formatted_date = expiration_date.strftime('%m/%d/%y')

                file_path = os.path.join(data_dir, filename)
                df = pd.read_csv(file_path)

                if "openInterest" not in df.columns or "strike" not in df.columns:
                    logging.warning(
                        f"Missing columns in {file_path}. Available columns: {df.columns}"
                    )

                sorted_data[formatted_date] = df

            except ValueError as e:
                logging.error(f"Error processing {filename}: {e}")

    # Sort by date keys (which are in mm/dd/yy format)
    return dict(
        sorted(
            sorted_data.items(),
            key=lambda x: datetime.strptime(x[0], '%m/%d/%y')
        )
    )


def fetch_stock_data(ticker, stock_obj=None):
    """
    Fetch current stock price data with retries.
    If stock_obj is provided, re-use that instead of creating a new one.
    """
    # Re-use existing Ticker object if available
    if stock_obj is None:
        stock_obj = yf.Ticker(ticker)

    for attempt in range(5):
        try:
            current_data = stock_obj.history(period="1d")
            if not current_data.empty:
                return current_data['Close'].iloc[-1]
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            time.sleep(2 ** attempt)

    return 0.0  # Default value if all attempts fail


def format_dollar_amount(amount):
    """
    Format numbers into a human-readable currency format.
    """
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.1f}K"
    else:
        return f"${amount:.2f}"


def gather_options_data(ticker, stock_obj=None):
    """
    Extract insights from the options chain data.
    :param ticker: Ticker symbol string
    :param stock_obj: Optional yf.Ticker object to re-use
    """
    ticker_path = os.path.join(TICKER_DIR, ticker)

    calls_folder = os.path.join(ticker_path, "CALLS")
    puts_folder = os.path.join(ticker_path, "PUTS")

    if not os.path.exists(calls_folder) or not os.path.exists(puts_folder):
        logging.warning(f"Missing data folders (CALLS/PUTS) for {ticker}. Skipping...")
        return {}

    # Preprocess calls/puts CSV files from the /shared_data folder
    calls_data = preprocess_dates(calls_folder, "CALLS")
    puts_data = preprocess_dates(puts_folder, "PUTS")

    calls_oi = {date: df['openInterest'].sum()
                for date, df in calls_data.items() if not df.empty}
    puts_oi = {date: df['openInterest'].sum()
               for date, df in puts_data.items() if not df.empty}
    calls_volume = {date: df['volume'].sum()
                    for date, df in calls_data.items() if not df.empty}
    puts_volume = {date: df['volume'].sum()
                   for date, df in puts_data.items() if not df.empty}

    logging.info(f"Processed OI -> Calls: {calls_oi}, Puts: {puts_oi}")
    logging.info(f"Processed Volume -> Calls: {calls_volume}, Puts: {puts_volume}")

    max_strike_calls, second_max_strike_calls, third_max_strike_calls = {}, {}, {}
    max_strike_puts, second_max_strike_puts, third_max_strike_puts = {}, {}, {}
    avg_strike = {}
    top_volume_contracts = []

    # If no existing stock object is passed, create a new one
    if stock_obj is None:
        stock_obj = yf.Ticker(ticker)

    # Fetch current stock price and company name
    try:
        current_price = fetch_stock_data(ticker, stock_obj)
        company_name = stock_obj.info.get('longName', 'N/A')
    except Exception as e:
        logging.error(f"Failed to fetch additional data for {ticker}: {e}")
        current_price, company_name = 0.0, "Unknown"

    def process_option_data(option_data, max_strike_dicts, option_type):
        """
        Go through each expiration date's DataFrame, find the highest open interest strikes,
        and the single contract with the highest volume.
        """
        for date, df in option_data.items():
            if df.empty:
                continue

            # Sort by open interest descending
            sorted_data = df.sort_values(by='openInterest', ascending=False)

            max_strike_dicts[0][date] = sorted_data.iloc[0]['strike'] if len(sorted_data) > 0 else 0
            max_strike_dicts[1][date] = sorted_data.iloc[1]['strike'] if len(sorted_data) > 1 else 0
            max_strike_dicts[2][date] = sorted_data.iloc[2]['strike'] if len(sorted_data) > 2 else 0

            # Identify the highest-volume contract
            if df['volume'].notna().any():
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

    # Process calls and puts
    process_option_data(calls_data, [max_strike_calls, second_max_strike_calls, third_max_strike_calls], "CALL")
    process_option_data(puts_data,  [max_strike_puts,  second_max_strike_puts,  third_max_strike_puts ], "PUT")

    # Calculate an "avg strike" for calls+puts combined (if you need it)
    for date in max_strike_calls.keys():
        # Ensure date is in puts too
        if date in max_strike_puts:
            total_calls_oi = calls_oi.get(date, 0)
            total_puts_oi = puts_oi.get(date, 0)
            total_oi = total_calls_oi + total_puts_oi
            if total_oi > 0:
                # Weighted by open interest
                avg_strike[date] = (
                    (max_strike_calls[date] * total_calls_oi +
                     max_strike_puts[date]  * total_puts_oi) / total_oi
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
