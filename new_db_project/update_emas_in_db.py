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

# ─── Single‐ticker EMA computation & update ────────────────────────────────
def compute_and_update_emas(symbol: str):
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
    if df.empty:
        print(f"⚠️ No data for {symbol}")
        return

    df.set_index("date", inplace=True)

    # compute each EMA
    for period in EMA_PERIODS:
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    update_sql = text(f"""
        UPDATE price_history
           SET {', '.join(f"ema_{p} = :ema_{p}" for p in EMA_PERIODS)}
         WHERE symbol = :sym
           AND date   = :dt
    """)

    # batch‐up updates in one transaction
    with engine.begin() as conn:
        for idx, row in df.iterrows():
            # cast to native Python types
            params = {
                **{
                    f"ema_{p}": float(row[f"ema_{p}"])
                    if pd.notna(row[f"ema_{p}"]) else None
                    for p in EMA_PERIODS
                },
                "sym": symbol,
                "dt": idx
            }
            conn.execute(update_sql, params)

    print(f"✅ {symbol}: EMAs updated ({len(df)} rows)")

# ─── Parallel runner ────────────────────────────────────────────────────────
def run_ema_update_all(max_workers: int = 8):
    # grab all distinct symbols
    symbols = pd.read_sql(
        text("SELECT DISTINCT symbol FROM price_history ORDER BY symbol"),
        engine
    )["symbol"].tolist()

    print(f"▶️  Computing EMAs for {len(symbols)} symbols with {max_workers} workers")

    # thread‐based pool (I/O + light CPU)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_sym = {pool.submit(compute_and_update_emas, sym): sym for sym in symbols}
        for fut in tqdm(as_completed(future_to_sym), total=len(future_to_sym), desc="EMAs"):
            sym = future_to_sym[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"❌ {sym} failed: {e}")

    print("🎉 All EMAs updated in parallel.")

# ─── Script entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    run_ema_update_all(max_workers=8)
