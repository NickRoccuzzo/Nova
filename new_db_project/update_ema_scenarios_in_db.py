import pandas as pd
from sqlalchemy import create_engine, text

# ─── DATABASE SETUP ─────────────────────────────────────────────────────────────
engine = create_engine(
    "postgresql://option_user:option_pass@localhost:5432/tickers",
    future=True
)

# ─── EMA SETTINGS ───────────────────────────────────────────────────────────────
EMA_PERIODS = [3, 5, 7, 9, 12, 15, 18, 21, 25, 29, 33, 37, 42, 47, 50, 52, 57, 75, 85, 95, 100, 105, 115, 125, 150, 200]

# ─── CURRENT SCENARIO DETECTION ─────────────────────────────────────────────────
def current_scenario_detector(symbol: str) -> str:
    """Return the current EMA scenario string, auto-stopping at the last valid EMA."""
    cols = ", ".join(f"ema_{p}" for p in EMA_PERIODS)
    df = pd.read_sql(
        text(f"""
        SELECT date, close, {cols}
          FROM price_history
         WHERE symbol = :sym
         ORDER BY date DESC
         LIMIT 1
        """),
        engine,
        params={"sym": symbol},
        parse_dates=["date"]
    )
    if df.empty:
        return "No data"

    row = df.iloc[0]
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

    scenario = " ".join(parts)
    print(f"Scenario for {symbol}: {scenario}")
    return scenario

# ─── AVERAGE RETURN CALCULATION ────────────────────────────────────────────────
def compute_avg_return_for_scenario(symbol: str, scenario: str, lookahead_days: int = 5) -> dict:
    """Compute avg return, bull%, bear%, and count for a given symbol's scenario."""
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
    scenarios = []
    for _, row in df.iterrows():
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
        scenarios.append(" ".join(parts))
    df["scenario"] = scenarios

    returns = []
    for i in range(len(df) - lookahead_days):
        if df.iloc[i]["scenario"] == scenario:
            now = float(df.iloc[i]["close"])
            future = float(df.iloc[i + lookahead_days]["close"])
            returns.append((future / now - 1) * 100)

    if not returns:
        print(f"No matches for scenario [{scenario}] on {symbol}")
        return {"avg_return": 0.0, "bull_%": 0.0, "bear_%": 0.0, "num_occurrences": 0}

    avg = float(sum(returns) / len(returns))
    bull = float(sum(1 for r in returns if r > 0) / len(returns) * 100)
    bear = float(sum(1 for r in returns if r < 0) / len(returns) * 100)
    count = int(len(returns))

    return {"avg_return": avg, "bull_%": bull, "bear_%": bear, "num_occurrences": count}

# ─── LATEST CROSS & MOMENTUM DETECTION ──────────────────────────────────────────
def detect_latest_cross_and_momentum(row):
    """
    Given one price_history row (with all ema_NNN columns),
    find the last valid EMA pair and return:
      - latest_cross (e.g. "150<200")
      - momentum ("BEARISH" or "BULLISH")
    """
    last_idx = -1
    for i, p in enumerate(EMA_PERIODS):
        if pd.isna(row[f"ema_{p}"]):
            break
        last_idx = i

    if last_idx < 1:
        return None, None

    prev_p = EMA_PERIODS[last_idx - 1]
    curr_p = EMA_PERIODS[last_idx]
    prev_val = row[f"ema_{prev_p}"]
    curr_val = row[f"ema_{curr_p}"]

    if curr_val > prev_val:
        op, mom = ">", "BULLISH"
    else:
        op, mom = "<", "BEARISH"

    return f"{prev_p}{op}{curr_p}", mom

# ─── UPDATE DB FIELDS ─────────────────────────────────────────────────────────
def update_current_scenario(symbol, lookahead_days=5):
    # 1) fetch the latest row
    row_df = pd.read_sql(
        text(f"""
        SELECT date, close, current_scenario,
               avg_return_for_scenario, bull_percent_for_scenario,
               bear_percent_for_scenario, num_occurrences_for_scenario,
               {', '.join(f'ema_{p}' for p in EMA_PERIODS)}
          FROM price_history
         WHERE symbol = :sym
         ORDER BY date DESC
         LIMIT 1
        """),
        engine,
        params={"sym": symbol},
        parse_dates=["date"]
    )
    if row_df.empty or pd.isna(row_df.iloc[0]["current_scenario"]):
        print(f"⚠️ No scenario data for {symbol}, skipping")
        return

    row = row_df.iloc[0]

    # 2) recompute avg & win/loss stats
    result = compute_avg_return_for_scenario(symbol, row["current_scenario"], lookahead_days)

    # 3) detect the latest EMA cross and momentum
    latest_cross, momentum = detect_latest_cross_and_momentum(row)

    # 4) write back to the DB
    with engine.begin() as conn:
        sql = text("""
            UPDATE price_history
               SET current_scenario             = :scenario,
                   avg_return_for_scenario      = :avg_return,
                   bull_percent_for_scenario    = :bull_percent,
                   bear_percent_for_scenario    = :bear_percent,
                   num_occurrences_for_scenario = :num_occurrences,
                   latest_cross                 = :latest_cross,
                   momentum                     = :momentum
             WHERE symbol = :sym
               AND date   = (
                   SELECT MAX(date) FROM price_history WHERE symbol = :sym
               )
        """)
        params = {
            "scenario":        row["current_scenario"],
            "avg_return":      float(result["avg_return"]),
            "bull_percent":    float(result["bull_%"]),
            "bear_percent":    float(result["bear_%"]),
            "num_occurrences": int(result["num_occurrences"]),
            "latest_cross":    latest_cross,
            "momentum":        momentum,
            "sym":             symbol
        }
        conn.execute(sql, params)

    print(f"✅ {symbol}: latest_cross={latest_cross}, momentum={momentum}")

# ─── BATCH RUNNER ──────────────────────────────────────────────────────────────
def run_update_all_scenarios(lookahead_days: int = 5):
    symbols = pd.read_sql(
        text("SELECT DISTINCT symbol FROM price_history ORDER BY symbol"),
        engine
    )["symbol"].tolist()

    print(f"Updating scenarios for {len(symbols)} symbols…")
    for sym in symbols:
        update_current_scenario(sym, lookahead_days)
    print("🎉 All scenarios updated.")

# ─── EXECUTE AS SCRIPT ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Make sure you’ve run your EMA builder first
    # 2) Then populate scenarios & stats:
    run_update_all_scenarios(lookahead_days=5)
