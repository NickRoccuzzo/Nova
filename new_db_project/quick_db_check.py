from sqlalchemy import create_engine, text
from db_config import POSTGRES_DB_URL

engine = create_engine(POSTGRES_DB_URL)

with engine.connect() as conn:
    summary = conn.execute(text("""
        SELECT
          COUNT(*)                 AS total,
          COUNT(last_queried_time) AS done,
          MIN(last_queried_time)   AS first,
          MAX(last_queried_time)   AS last
        FROM tickers
    """)).fetchone()
    print("Tickers:", summary)

    top5 = conn.execute(text("""
        SELECT t.symbol, COUNT(*) AS rows
        FROM option_contracts oc
        JOIN tickers t ON t.ticker_id = oc.ticker_id
        GROUP BY t.symbol
        ORDER BY rows DESC
        LIMIT 5
    """)).fetchall()
    print("Top 5 row counts:", top5)

    # Now check the live metrics:
    metrics = conn.execute(text("""
        SELECT ticker_id, call_vol_total, put_vol_total
        FROM ticker_metrics
        ORDER BY ticker_id
        LIMIT 5
    """)).fetchall()
    print("Ticker metrics snippet:", metrics)