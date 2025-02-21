#!/usr/bin/env python3
"""
NovaPyLogic.py - Updated Version

This module processes raw options data (stored as CSVs) to extract insights.
An asynchronous wrapper for fetching stock data is provided to support
concurrent network calls when used with asyncio.
"""

import os
import re
from pathlib import Path
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import yfinance as yf

# For async support
import asyncio

# Configure logging
logging.basicConfig(
    filename="novapylogic.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Shared storage directory (from Docker ENV or default)
TICKER_DIR = Path(os.getenv("TICKER_DIR", "/shared_data"))


def preprocess_dates(data_dir: Path, file_suffix: str) -> Dict[str, pd.DataFrame]:
    """
    Preprocess CSV files for options data by extracting and sorting expiration dates.

    Filenames should match the pattern: YYYYMMDD_{file_suffix}.csv.
    """
    sorted_data: Dict[str, pd.DataFrame] = {}
    if not data_dir.exists():
        logging.warning(f"Directory {data_dir} does not exist. Skipping...")
        return sorted_data

    pattern = re.compile(r"(\d{8})_" + re.escape(file_suffix) + r"\.csv$", re.IGNORECASE)
    for file_path in data_dir.glob("*.csv"):
        match = pattern.search(file_path.name)
        if match:
            date_str = match.group(1)
            try:
                expiration_date = datetime.strptime(date_str, '%Y%m%d')
                formatted_date = expiration_date.strftime('%m/%d/%y')
                df = pd.read_csv(file_path)
                # Warn if key columns are missing
                for col in ["openInterest", "strike"]:
                    if col not in df.columns:
                        logging.warning(
                            f"Missing '{col}' column in {file_path}. Available columns: {list(df.columns)}"
                        )
                sorted_data[formatted_date] = df
            except ValueError as e:
                logging.error(f"Error processing {file_path.name}: {e}", exc_info=True)
    return dict(sorted(sorted_data.items(), key=lambda x: datetime.strptime(x[0], '%m/%d/%y')))


def fetch_stock_data(ticker: str, stock_obj: Optional[yf.Ticker] = None) -> float:
    """
    Fetch the current closing price for a stock, with retry logic.
    This is the synchronous version.
    """
    if stock_obj is None:
        stock_obj = yf.Ticker(ticker)
    for attempt in range(5):
        try:
            current_data = stock_obj.history(period="1d")
            if not current_data.empty:
                return float(current_data['Close'].iloc[-1])
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {ticker}: {e}", exc_info=True)
            time.sleep(2 ** attempt)
    logging.error(f"All attempts failed for fetching data for {ticker}. Returning 0.0.")
    return 0.0


async def async_fetch_stock_data(ticker: str, stock_obj: Optional[yf.Ticker] = None) -> float:
    """
    Asynchronously fetch the current closing price for a stock.
    This wraps the blocking stock.history call in run_in_executor.
    """
    loop = asyncio.get_running_loop()
    if stock_obj is None:
        stock_obj = await loop.run_in_executor(None, lambda: yf.Ticker(ticker))
    for attempt in range(5):
        try:
            current_data = await loop.run_in_executor(None, stock_obj.history, "1d")
            if not current_data.empty:
                return float(current_data['Close'].iloc[-1])
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {ticker} (async): {e}", exc_info=True)
            await asyncio.sleep(2 ** attempt)
    logging.error(f"All async attempts failed for {ticker}. Returning 0.0.")
    return 0.0


def format_dollar_amount(amount: float) -> str:
    """
    Format a numeric value into a human-readable dollar amount.
    """
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.1f}K"
    else:
        return f"${amount:.2f}"


def process_option_data(
    option_data: Dict[str, pd.DataFrame],
    option_type: str,
    top_volume_contracts: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Process option data to extract the top three strikes by open interest and identify
    the contract with the highest volume for each expiration date.
    """
    strikes = {"max": {}, "second": {}, "third": {}}
    for date, df in option_data.items():
        if df.empty:
            continue
        try:
            sorted_data = df.sort_values(by='openInterest', ascending=False).reset_index(drop=True)
            strikes["max"][date] = sorted_data.at[0, 'strike'] if len(sorted_data) > 0 else 0
            strikes["second"][date] = sorted_data.at[1, 'strike'] if len(sorted_data) > 1 else 0
            strikes["third"][date] = sorted_data.at[2, 'strike'] if len(sorted_data) > 2 else 0
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
        except Exception as e:
            logging.error(f"Error processing option data for {date} ({option_type}): {e}", exc_info=True)
    return strikes


def gather_options_data(ticker: str, stock_obj: Optional[yf.Ticker] = None) -> Dict[str, Any]:
    """
    Extract insights from stored options data for the specified ticker.
    """
    ticker_path = TICKER_DIR / ticker
    calls_folder = ticker_path / "CALLS"
    puts_folder = ticker_path / "PUTS"
    if not calls_folder.exists() or not puts_folder.exists():
        logging.warning(f"Missing data folders (CALLS/PUTS) for {ticker}. Skipping...")
        return {}

    # Preprocess CSV files
    calls_data = preprocess_dates(calls_folder, "CALLS")
    puts_data = preprocess_dates(puts_folder, "PUTS")

    calls_oi = {date: df['openInterest'].sum() for date, df in calls_data.items() if not df.empty}
    puts_oi = {date: df['openInterest'].sum() for date, df in puts_data.items() if not df.empty}
    calls_volume = {date: df['volume'].sum() for date, df in calls_data.items() if not df.empty}
    puts_volume = {date: df['volume'].sum() for date, df in puts_data.items() if not df.empty}

    logging.info(f"Processed OI -> Calls: {calls_oi}, Puts: {puts_oi}")
    logging.info(f"Processed Volume -> Calls: {calls_volume}, Puts: {puts_volume}")

    top_volume_contracts: List[Dict[str, Any]] = []

    if stock_obj is None:
        stock_obj = yf.Ticker(ticker)

    try:
        # You can choose to use the synchronous or async version here.
        current_price = fetch_stock_data(ticker, stock_obj)
        company_name = stock_obj.info.get('longName', 'N/A')
    except Exception as e:
        logging.error(f"Failed to fetch additional data for {ticker}: {e}", exc_info=True)
        current_price, company_name = 0.0, "Unknown"

    calls_strikes = process_option_data(calls_data, "CALL", top_volume_contracts)
    puts_strikes = process_option_data(puts_data, "PUT", top_volume_contracts)

    avg_strike = {}
    for date in calls_strikes.get("max", {}):
        if date in puts_strikes.get("max", {}):
            total_calls_oi = calls_oi.get(date, 0)
            total_puts_oi = puts_oi.get(date, 0)
            total_oi = total_calls_oi + total_puts_oi
            if total_oi > 0:
                avg_strike[date] = (calls_strikes["max"][date] * total_calls_oi +
                                    puts_strikes["max"][date] * total_puts_oi) / total_oi
            else:
                avg_strike[date] = np.nan

    return {
        "calls_oi": calls_oi,
        "puts_oi": puts_oi,
        "calls_volume": calls_volume,
        "puts_volume": puts_volume,
        "max_strike_calls": calls_strikes.get("max", {}),
        "second_max_strike_calls": calls_strikes.get("second", {}),
        "third_max_strike_calls": calls_strikes.get("third", {}),
        "max_strike_puts": puts_strikes.get("max", {}),
        "second_max_strike_puts": puts_strikes.get("second", {}),
        "third_max_strike_puts": puts_strikes.get("third", {}),
        "avg_strike": avg_strike,
        "top_volume_contracts": top_volume_contracts,
        "current_price": current_price,
        "company_name": company_name
    }
