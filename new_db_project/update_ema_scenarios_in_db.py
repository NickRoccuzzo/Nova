import pandas as pd
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DB_URL       = "postgresql://option_user:option_pass@localhost:5432/tickers"
EMA_PERIODS  = [3, 5, 7, 9, 12, 15, 18, 21, 25, 29, 33, 37, 42, 47, 50,
                52, 57, 75, 85, 95, 100, 105, 115, 125, 150, 200]
LOOKAHEAD_DAYS = 5
MAX_WORKERS    = 8

# ─── ENGINE ───────────────────────────────────────────────────────────────────
engine = create_engine(DB_URL, future=True)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def scenario_from_row(row):
    parts = []
    for i, p in enumerate(EMA_PERIODS):
        val = row[f"ema_{p}"]
        if pd.isna(val):
            break
        if i == 0:
            parts.append(str(p))
        else:
            prev = row[f"ema_{EMA_PERIODS[i-1]}"]
            if pd.isna(prev):
                break
            parts.append(("> " if val > prev else "< ") + str(p))
    return " ".join(parts)

def detect_latest_cross_and_momentum(row):
    last_idx = -1
    for i, p in enumerate(EMA_PERIODS):
        if pd.isna(row[f"ema_{p}"]):
            break
        last_idx = i
    if last_idx < 1:
        return None, None
    prev_p, curr_p = EMA_PERIODS[last_idx-1], EMA_PERIODS[last_idx]
    prev_val, curr_val = row[f"ema_{prev_p}"], row[f"ema_{curr_p}"]
    if curr_val > prev_val:
        return f"{prev_p}>{curr_p}", "BULLISH"
    else:
        return f"{prev_p}<{curr_p}", "BEARISH"

def compute_avg_return_for_scenario(symbol: str, scenario: str, lookahead_days: int):
    cols = ", ".join(f"ema_{p}" for p in EMA_PERIODS)
    df = pd.read_sql(
        text(f"""
        SELECT date, close, {cols}
          FROM price_history
         WHERE symbol = :sym
         ORDER BY date ASC
        """),
        engine,
        params={"sym": symbol},
        parse_dates=["date"]
    )
    if df.empty:
        return {"avg_return": 0.0, "bull_%": 0.0, "bear_%": 0.0, "num_occurrences": 0}

    df.set_index("date", inplace=True)
    # build scenario history
    history = []
    for _, row in df.iterrows():
        history.append(scenario_from_row(row))
    df["scenario"] = history

    returns = []
    for i in range(len(df) - lookahead_days):
        if df.iloc[i]["scenario"] == scenario:
            now, future = float(df.iloc[i]["close"]), float(df.iloc[i + lookahead_days]["close"])
            returns.append((future / now - 1) * 100)

    if not returns:
        return {"avg_return": 0.0, "bull_%": 0.0, "bear_%": 0.0, "num_occurrences": 0}

    avg    = float(sum(returns) / len(returns))
    bull   = float(sum(1 for r in returns if r > 0) / len(returns) * 100)
    bear   = float(sum(1 for r in returns if r < 0) / len(returns) * 100)
    count  = len(returns)
    return {"avg_return": avg, "bull_%": bull, "bear_%": bear, "num_occurrences": count}

def compute_scenario_params(symbol: str):
    # 1) grab the very latest row of price + EMAs
    row_df = pd.read_sql(
        text(f"""
        SELECT date, close, {', '.join(f'ema_{p}' for p in EMA_PERIODS)}
          FROM price_history
         WHERE symbol = :sym
         ORDER BY date DESC
         LIMIT 1
        """),
        engine,
        params={"sym": symbol},
        parse_dates=["date"]
    )
    if row_df.empty:
        print(f"⚠️ No data for {symbol}")
        return None

    row = row_df.iloc[0]
    dt  = row["date"]

    # 2) compute the current scenario & stats
    scenario = scenario_from_row(row)
    stats    = compute_avg_return_for_scenario(symbol, scenario, LOOKAHEAD_DAYS)
    cross, mom = detect_latest_cross_and_momentum(row)

    return {
        "sym":             symbol,
        "dt":              dt,
        "scenario":        scenario,
        "avg_return":      float(stats["avg_return"]),
        "bull_percent":    float(stats["bull_%"]),
        "bear_percent":    float(stats["bear_%"]),
        "num_occurrences": int(stats["num_occurrences"]),
        "latest_cross":    cross,
        "momentum":        mom
    }

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run_update_all_scenarios():
    # A) detect which symbols need re-running
    df_dates = pd.read_sql(
        text("""
        SELECT symbol,
               MAX(date)                         AS price_date,
               MAX(date) FILTER (WHERE current_scenario IS NOT NULL) AS scen_date
          FROM price_history
         GROUP BY symbol
        """),
        engine,
        parse_dates=["price_date", "scen_date"]
    )

    # ← updated here
    df_dates["scen_date"] = df_dates["scen_date"].fillna(pd.Timestamp("1900-01-01"))

    to_process = df_dates.loc[df_dates["price_date"] > df_dates["scen_date"], "symbol"].tolist()

    if not to_process:
        print("✅ No symbols need scenario updates.")
        return

    print(f"▶️  Updating scenarios for {len(to_process)} symbols…")

    # B) parallel compute all params
    params_list = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(compute_scenario_params, s): s for s in to_process}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scenarios"):
            sym = futures[fut]
            try:
                result = fut.result()
                if result:
                    params_list.append(result)
                    print(f"✔️ {sym} computed")
            except Exception as e:
                print(f"❌ {sym} failed: {e}")

    # C) batch-update into price_history
    update_sql = text(f"""
    UPDATE price_history
       SET current_scenario             = :scenario,
           avg_return_for_scenario      = :avg_return,
           bull_percent_for_scenario    = :bull_percent,
           bear_percent_for_scenario    = :bear_percent,
           num_occurrences_for_scenario = :num_occurrences,
           latest_cross                 = :latest_cross,
           momentum                     = :momentum
     WHERE symbol = :sym
       AND date   = :dt
    """)
    with engine.begin() as conn:
        conn.execute(update_sql, params_list)

    print("🎉 All scenarios updated incrementally.")

if __name__ == "__main__":
    run_update_all_scenarios()
