import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import pandas as pd
from tqdm import tqdm

BASE_DIR     = "tickers_pricehistory"
TICKERS_JSON = "tickers.json"

def load_tickers():
    with open(TICKERS_JSON, "r") as f:
        return json.load(f)

def create_folder_structure(tickers_map):
    for sector, industries in tickers_map.items():
        for industry in industries:
            path = os.path.join(BASE_DIR, sector, industry)
            os.makedirs(path, exist_ok=True)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # drop unwanted columns
    df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
    # reformat index to "MM-DD-YYYY"
    df.index = pd.to_datetime(df.index).strftime("%m-%d-%Y")
    # round price columns to 3 decimals
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col].round(3)
    # drop any rows with NaNs
    return df.dropna()

def fetch_history(ticker: str) -> pd.DataFrame:
    for attempt in range(1, 4):
        try:
            return yf.Ticker(ticker).history(period="max")
        except Exception:
            if attempt < 3:
                time.sleep(3)
            else:
                raise

def download_ticker(ticker: str, sector: str, industry: str) -> bool:
    try:
        df = fetch_history(ticker)
        df = clean_dataframe(df)
        out_path = os.path.join(BASE_DIR, sector, industry, f"{ticker}.csv")
        df.to_csv(out_path)
        return True
    except Exception:
        return False

def find_sector_industry(tickers_map, ticker):
    for sector, industries in tickers_map.items():
        for industry, tickers in industries.items():
            if ticker in tickers:
                return sector, industry
    return None, None

def download_price_history():
    tickers_map = load_tickers()
    create_folder_structure(tickers_map)

    # build list of all (ticker, sector, industry)
    tasks = [
        (tkr, sec, ind)
        for sec, inds in tickers_map.items()
        for ind, tl in inds.items()
        for tkr in tl
    ]

    # first pass
    failures = []
    with ThreadPoolExecutor(max_workers=2) as exe:
        futures = {
            exe.submit(download_ticker, tkr, sec, ind): tkr
            for tkr, sec, ind in tasks
        }
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        ncols=60,
                        desc="Downloading"):
            tkr = futures[fut]
            if not fut.result():
                failures.append(tkr)

    # retry pass
    if failures:
        retry_failures = []
        with ThreadPoolExecutor(max_workers=2) as exe:
            futures = {
                exe.submit(
                    download_ticker,
                    tkr,
                    *find_sector_industry(tickers_map, tkr)
                ): tkr
                for tkr in failures
            }
            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            ncols=60,
                            desc="Retrying"):
                tkr = futures[fut]
                if not fut.result():
                    retry_failures.append(tkr)

        if retry_failures:
            print("Failed after retries:", retry_failures)

def folder_synchronizer():
    tickers_map = load_tickers()
    create_folder_structure(tickers_map)
    today = pd.Timestamp.today().strftime("%m-%d-%Y")

    full_fetch = []
    to_update = []

    # determine which tickers need full history vs daily append
    for sec, inds in tickers_map.items():
        for ind, tl in inds.items():
            for tkr in tl:
                csv_path = os.path.join(BASE_DIR, sec, ind, f"{tkr}.csv")
                if os.path.exists(csv_path):
                    to_update.append((tkr, sec, ind, csv_path))
                else:
                    full_fetch.append((tkr, sec, ind))

    # full history for new tickers
    if full_fetch:
        with ThreadPoolExecutor(max_workers=2) as exe:
            futures = {
                exe.submit(download_ticker, tkr, sec, ind): tkr
                for tkr, sec, ind in full_fetch
            }
            for _ in tqdm(as_completed(futures),
                          total=len(futures),
                          ncols=60,
                          desc="Syncing new"):
                pass

    # append today's row for existing tickers
    for tkr, sec, ind, path in tqdm(to_update,
                                   ncols=60,
                                   desc="Updating today"):
        try:
            df_existing = pd.read_csv(path, index_col=0, parse_dates=False)
            if today not in df_existing.index:
                df_new = clean_dataframe(fetch_history(tkr))
                if today in df_new.index:
                    df_new.loc[today].to_frame().T.to_csv(
                        path, mode="a", header=False
                    )
        except Exception:
            continue

if __name__ == "__main__":
    # either run a full fetch:
    # download_price_history()
    # — or —
    # keep your folder in sync:
     folder_synchronizer()
