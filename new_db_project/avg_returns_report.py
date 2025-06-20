#!/usr/bin/env python3
import pandas as pd
from sqlalchemy import create_engine, text

# ─── Configuration ───────────────────────────────────────────────────────────
DB_URL = "postgresql://option_user:option_pass@localhost:5432/tickers"

def avg_returns_report():
    """Connects to the DB, pulls the latest scenario stats per ticker, and prints a sorted report."""
    engine = create_engine(DB_URL, future=True)
    # Query the most recent stats for each symbol
    query = """
    SELECT ph.symbol,
           ph.avg_return_for_scenario AS avg_return_pct,
           ph.bull_percent_for_scenario AS bull_pct,
           ph.bear_percent_for_scenario AS bear_pct
      FROM price_history ph
      JOIN (
        SELECT symbol, MAX(date) AS max_date
          FROM price_history
         GROUP BY symbol
      ) recent
        ON ph.symbol = recent.symbol
       AND ph.date   = recent.max_date
     ORDER BY ph.avg_return_for_scenario DESC;
    """
    df = pd.read_sql(text(query), engine)
    if df.empty:
        print("No data found in price_history.")
        return

    # Display the report
    print("Ticker  |  Avg Return (%)  |  Bull %  |  Bear %")
    print("-" * 50)
    for _, row in df.iterrows():
        print(f"{row['symbol']:6} | {row['avg_return_pct']:16.2f} | {row['bull_pct']:7.2f}% | {row['bear_pct']:7.2f}%")

if __name__ == "__main__":
    avg_returns_report()
