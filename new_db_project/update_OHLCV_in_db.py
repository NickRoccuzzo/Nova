import time
import random
from itertools import islice

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
import sqlalchemy.types

# ─── Configuration ────────────────────────────────────────────────────────
DB_URL        = "postgresql://option_user:option_pass@localhost:5432/tickers"
TICKERS_TABLE = "tickers"
PRICE_TABLE   = "price_history"
BATCH_SIZE    = 5  # how many tickers to fetch in one yf.download call

# ─── Helper Functions ─────────────────────────────────────────────────────

def chunked(iterable, size):
    """Yield successive chunks of length size."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]

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
    all_tickers = get_all_tickers(engine)

    # Partition into existing vs. brand-new symbols
    existing = {}
    new      = []
    for sym in all_tickers:
        ld = get_last_date(engine, sym)
        if ld:
            existing[sym] = ld
        else:
            new.append(sym)

    # 1) Process existing tickers in batches
    for batch in chunked(list(existing.keys()), BATCH_SIZE):
        # compute the earliest start date among the batch
        next_starts = [existing[sym] + pd.Timedelta(days=1) for sym in batch]
        start_date  = min(next_starts)

        # batch download
        df_all = yf.download(
            batch,
            start=start_date,
            group_by="ticker",
            auto_adjust=False
        )

        batch_dfs = []
        updates   = []

        for sym in batch:
            # extract per-symbol DataFrame (fallback if group_by didn’t nest)
            try:
                df = df_all[sym]
            except Exception:
                df = df_all.copy()

            # flatten MultiIndex if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)

            df = clean_dataframe(df)
            df["symbol"] = sym
            # filter out anything ≤ last_date
            df = df[df["date"] > pd.to_datetime(existing[sym])]

            if df.empty:
                print(f"⚠️ {sym} already up to date through {existing[sym]}")
                continue

            batch_dfs.append(df)
            updates.append((sym, df["date"].max().date()))

        # one bulk insert per batch
        if batch_dfs:
            to_insert = pd.concat(batch_dfs, ignore_index=True)
            to_insert.to_sql(
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
            for sym, up_to in updates:
                print(f"✔️ {sym} updated through {up_to}")

        # avoid hammering the API
        time.sleep(random.uniform(1, 3))

    # 2) Process brand-new tickers (full history)
    for sym in new:
        df = fetch_history(sym)
        if df is None or df.empty:
            print(f"⚠️ No data for {sym}")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)

        df = clean_dataframe(df)
        df["symbol"] = sym

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
        print(f"✔️ {sym} full history inserted")

if __name__ == "__main__":
    price_check()
