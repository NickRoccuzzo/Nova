import time

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
import sqlalchemy.types

# ─── Configuration ────────────────────────────────────────────────────────
DB_URL        = "postgresql://option_user:option_pass@localhost:5432/tickers"
TICKERS_TABLE = "tickers"
PRICE_TABLE   = "price_history"

# ─── Helper Functions ─────────────────────────────────────────────────────

def fetch_history(ticker: str) -> pd.DataFrame:
    """Download full history, with up to 3 retries."""
    for attempt in range(1, 4):
        try:
            return yf.Ticker(ticker).history(period="max")
        except Exception:
            if attempt < 3:
                time.sleep(3)
            else:
                raise

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unwanted cols, normalize names, round prices, drop NaNs."""
    df = df.drop(columns=["Dividends", "Stock Splits", "Adj Close"], errors="ignore")
    df = df.reset_index().rename(columns={"Date": "date"})
    for orig, new in [("Open", "open"), ("High", "high"),
                      ("Low",  "low"), ("Close", "close")]:
        if orig in df:
            df[new] = df[orig].round(3)
            df.drop(columns=[orig], inplace=True)
    if "Volume" in df:
        df = df.rename(columns={"Volume": "volume"})
    keep = {"date", "open", "high", "low", "close", "volume"}
    return df[[c for c in df.columns if c in keep]].dropna()

def ensure_price_table(engine):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {PRICE_TABLE} (
      symbol TEXT    NOT NULL,
      date   DATE    NOT NULL,
      open   NUMERIC,
      high   NUMERIC,
      low    NUMERIC,
      close  NUMERIC,
      volume BIGINT,
      PRIMARY KEY(symbol, date)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

def get_all_tickers(engine):
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT symbol FROM {TICKERS_TABLE}"))
        return [row[0] for row in result]

def get_last_date(engine, symbol):
    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT MAX(date) FROM {PRICE_TABLE} WHERE symbol = :symbol"),
            {"symbol": symbol}
        ).one()
        return row[0]

# ─── Main Workflow ────────────────────────────────────────────────────────

def price_check():
    engine = create_engine(DB_URL)
    ensure_price_table(engine)
    tickers = get_all_tickers(engine)

    for symbol in tickers:
        last_date = get_last_date(engine, symbol)

        # fetch only new data if we have a last date
        if last_date:
            df = yf.download(
                symbol,
                start=last_date + pd.Timedelta(days=1),
                group_by="column",    # flatten multi-index
                auto_adjust=False     # avoid future‐warning
            )
        else:
            df = fetch_history(symbol)

        if df is None or df.empty:
            print(f"⚠️ No data for {symbol}")
            continue

        # flatten any remaining MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)

        df = clean_dataframe(df)
        df["symbol"] = symbol

        # drop rows that are already in the DB
        if last_date is not None:
            # convert last_date (a Python date) to a pandas Timestamp
            last_ts = pd.to_datetime(last_date)
            df = df[df["date"] > last_ts]
            if df.empty:
                print(f"⚠️ {symbol} already up to date through {last_date}")
                continue

        # append new rows
        df.to_sql(
            PRICE_TABLE,
            engine,
            if_exists="append",
            index=False,
            dtype={
                "symbol": sqlalchemy.types.Text(),
                "date":   sqlalchemy.types.Date(),
                "open":   sqlalchemy.types.Numeric(),
                "high":   sqlalchemy.types.Numeric(),
                "low":    sqlalchemy.types.Numeric(),
                "close":  sqlalchemy.types.Numeric(),
                "volume": sqlalchemy.types.BigInteger(),
            }
        )
        print(f"✔️ {symbol} updated through {df['date'].max().date()}")
        time.sleep(1)

if __name__ == "__main__":
    price_check()
