import pandas as pd
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────
DB_URL      = "postgresql://option_user:option_pass@localhost:5432/tickers"
EMA_PERIODS = [3, 5, 7, 9, 12, 15, 18, 21, 25, 29, 33, 37, 42, 47, 50,
               52, 57, 75, 85, 95, 100, 105, 115, 125, 150, 200]

# Create a single Engine; SQLAlchemy pools safely across threads
engine = create_engine(DB_URL, future=True)


def ensure_ema_columns():
    """Auto-add any missing ema_<period> DOUBLE PRECISION columns."""
    with engine.begin() as conn:
        existing = {
            row[0]
            for row in conn.execute(text("""
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_name = 'price_history'
            """))
        }
        for p in EMA_PERIODS:
            col = f"ema_{p}"
            if col not in existing:
                conn.execute(text(f"""
                    ALTER TABLE price_history
                      ADD COLUMN IF NOT EXISTS {col}
                        DOUBLE PRECISION
                """))


def compute_and_update_emas(symbol: str):
    # 1) Find last EMA date (any period) we processed for this symbol
    with engine.connect() as conn:
        last_date = conn.execute(
            text(f"""
                SELECT MAX(date)
                  FROM price_history
                 WHERE symbol = :sym
                   AND ema_{EMA_PERIODS[0]} IS NOT NULL
            """),
            {"sym": symbol}
        ).scalar_one()

    # 2) Fetch seed EMA values and any new price rows
    if last_date is None:
        # no EMA at all — do full-history compute
        df = pd.read_sql(
            text("""
                SELECT date, close
                  FROM price_history
                 WHERE symbol = :sym
                 ORDER BY date ASC
            """),
            engine,
            params={"sym": symbol},
            parse_dates=["date"]
        )
        seed = {}
    else:
        # grab the existing EMA values at last_date
        cols = ", ".join(f"ema_{p}" for p in EMA_PERIODS)
        with engine.connect() as conn:
            seed_row = conn.execute(
                text(f"""
                    SELECT {cols}
                      FROM price_history
                     WHERE symbol = :sym AND date = :dt
                """),
                {"sym": symbol, "dt": last_date}
            ).one()
        seed = {EMA_PERIODS[i]: seed_row[i] for i in range(len(EMA_PERIODS))}

        # fetch only new prices
        df = pd.read_sql(
            text("""
                SELECT date, close
                  FROM price_history
                 WHERE symbol = :sym
                   AND date > :dt
                 ORDER BY date ASC
            """),
            engine,
            params={"sym": symbol, "dt": last_date},
            parse_dates=["date"]
        )

    if df.empty:
        print(f"⚠️ No new price data for {symbol}")
        return

    df.set_index("date", inplace=True)

    # 3) Compute incremental EMAs
    alpha = {p: 2.0 / (p + 1) for p in EMA_PERIODS}
    # Seed with first close if no prior EMA
    ema_values = {
        p: seed.get(p, float(df["close"].iloc[0]))
        for p in EMA_PERIODS
    }
    updates = []  # list of dict(params) to executemany

    for dt, row in df.iterrows():
        price = float(row["close"])
        params = {"sym": symbol, "dt": dt}
        for p in EMA_PERIODS:
            prev = ema_values[p]
            today = alpha[p] * price + (1 - alpha[p]) * prev
            ema_values[p] = today
            params[f"ema_{p}"] = today
        updates.append(params)

    # 4) Bulk UPDATE in one transaction
    set_clause = ", ".join(f"ema_{p} = :ema_{p}" for p in EMA_PERIODS)
    update_sql = text(f"""
        UPDATE price_history
           SET {set_clause}
         WHERE symbol = :sym
           AND date   = :dt
    """)
    with engine.begin() as conn:
        conn.execute(update_sql, updates)

    print(f"✅ {symbol}: EMAs incremental-updated ({len(updates)} rows)")


def run_ema_update_all(max_workers: int = 8):
    # 0) Auto-migrate columns
    ensure_ema_columns()

    # 1) Grab symbols to process
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT DISTINCT symbol FROM price_history ORDER BY symbol"
        ))
        symbols = [row[0] for row in result]

    print(f"▶️  Incremental EMAs for {len(symbols)} symbols with {max_workers} workers")

    # 2) Parallel runner
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(compute_and_update_emas, sym): sym for sym in symbols}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="EMAs"):
            sym = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"❌ {sym} failed: {e}")

    print("🎉 All EMAs updated incrementally.")


if __name__ == "__main__":
    run_ema_update_all(max_workers=8)
