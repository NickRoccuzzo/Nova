# -- MODULES -- #
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

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
            date_str = filename.split(file_suffix)[0]
            try:
                expiration_date = datetime.strptime(date_str, '%Y%m%d')
                formatted_date = expiration_date.strftime('%m/%d/%y')

                df = pd.read_csv(os.path.join(data_dir, filename))
                if not df.empty:
                    sorted_data[formatted_date] = df

            except ValueError as e:
                logging.error(f"Error processing {filename}: {e}")

    return dict(sorted(sorted_data.items(), key=lambda x: datetime.strptime(x[0], '%m/%d/%y')))


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

    calls_data = preprocess_dates(calls_dir, "CALLS")
    puts_data = preprocess_dates(puts_dir, "PUTS")

    calls_oi = {date: df['openInterest'].sum() for date, df in calls_data.items() if not df.empty}
    puts_oi = {date: df['openInterest'].sum() for date, df in puts_data.items() if not df.empty}

    max_strike_calls, second_max_strike_calls, third_max_strike_calls = {}, {}, {}
    max_strike_puts, second_max_strike_puts, third_max_strike_puts = {}, {}, {}

    top_volume_contracts = []

    # Fetch current stock price
    stock = yf.Ticker(ticker)
    current_data = stock.history(period="1d")
    current_price = current_data['Close'].iloc[-1] if not current_data.empty else 0.0
    company_name = stock.info.get('longName', 'N/A')

    def process_option_data(option_data, max_strike_dict, second_strike_dict, third_strike_dict):
        for date, df in option_data.items():
            if not df.empty:
                sorted_data = df.sort_values(by='openInterest', ascending=False)

                if len(sorted_data) > 0:
                    max_strike_dict[date] = sorted_data.iloc[0]['strike']
                else:
                    max_strike_dict[date] = 0

                if len(sorted_data) > 1:
                    second_strike_dict[date] = sorted_data.iloc[1]['strike']
                else:
                    second_strike_dict[date] = 0

                if len(sorted_data) > 2:
                    third_strike_dict[date] = sorted_data.iloc[2]['strike']
                else:
                    third_strike_dict[date] = 0

                # Find highest volume contract
                if 'volume' in df.columns and df['volume'].notna().any():
                    highest_volume_contract = df.loc[df['volume'].idxmax()]
                    total_spent = highest_volume_contract['volume'] * highest_volume_contract['lastPrice'] * 100
                    formatted_spent = format_dollar_amount(total_spent)

                    unusual = highest_volume_contract['volume'] > highest_volume_contract['openInterest']

                    top_volume_contracts.append({
                        'type': 'CALL' if file_suffix == "CALLS" else 'PUT',
                        'strike': highest_volume_contract['strike'],
                        'volume': highest_volume_contract['volume'],
                        'openInterest': highest_volume_contract['openInterest'],
                        'date': date,
                        'total_spent': formatted_spent,
                        'unusual': unusual
                    })

    # Process Calls and Puts
    process_option_data(calls_data, max_strike_calls, second_max_strike_calls, third_max_strike_calls)
    process_option_data(puts_data, max_strike_puts, second_max_strike_puts, third_max_strike_puts)

    return {
        "calls_oi": calls_oi,
        "puts_oi": puts_oi,
        "max_strike_calls": max_strike_calls,
        "second_max_strike_calls": second_max_strike_calls,
        "third_max_strike_calls": third_max_strike_calls,
        "max_strike_puts": max_strike_puts,
        "second_max_strike_puts": second_max_strike_puts,
        "third_max_strike_puts": third_max_strike_puts,
        "top_volume_contracts": top_volume_contracts,
        "current_price": current_price,
        "company_name": company_name
    }


def format_dollar_amount(amount):
    """
    Format a given dollar amount into a human-readable string with suffixes like 'K' for thousands and 'M' for millions.
    """
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.1f}K"
    else:
        return f"${amount:.2f}"


def process_tickers_in_parallel(tickers):
    """
    Runs gather_options_data() in parallel for multiple tickers.
    """
    logging.info(f"Processing {len(tickers)} tickers in parallel...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(gather_options_data, tickers))

    logging.info("Parallel processing complete.")
    return results


if __name__ == "__main__":
    SECTOR = os.getenv("SECTOR")

    if not SECTOR:
        raise ValueError("SECTOR environment variable is not set.")

    with open("tickers.json", "r") as f:
        tickers_data = json.load(f)

    if SECTOR not in tickers_data:
        raise ValueError(f"Sector '{SECTOR}' not found.")

    tickers = [ticker for industry in tickers_data[SECTOR].values() for ticker in industry]

    process_tickers_in_parallel(tickers)
