import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://option_user:option_pass@localhost:5432/tickers"

def unusual_volume_report():
    engine = create_engine(DATABASE_URL)

    # 1) Pull option_metrics + overall ticker_metrics
    df = pd.read_sql(text("""
    SELECT
      om.ticker_id,
      t.symbol,
      e.expiration_date,

      -- per-expiry totals & top-3 slots
      om.call_vol_sum,    om.max_vol_call,    om.max_call_oi,    om.max_call_strike,
      om.second_vol_call, om.second_call_oi,  om.second_call_strike,
      om.third_vol_call,  om.third_call_oi,   om.third_call_strike,

      om.put_vol_sum,     om.max_vol_put,     om.max_put_oi,     om.max_put_strike,
      om.second_vol_put,  om.second_put_oi,   om.second_put_strike,
      om.third_vol_put,   om.third_put_oi,    om.third_put_strike,

      -- global totals from your materialized view
      tm.call_vol_total,
      tm.put_vol_total

    FROM option_metrics om
    JOIN tickers t     USING(ticker_id)
    JOIN expirations e USING(expiration_id)
    JOIN ticker_metrics tm USING(ticker_id)
    """), engine, parse_dates=["expiration_date"])

    if df.empty:
        print("No data found.")
        return

    # 2) Compute normalized metrics
    df["call_vol_to_oi"] = df["max_vol_call"] / df["max_call_oi"].replace({0: np.nan})
    df["put_vol_to_oi"]  = df["max_vol_put"]  / df["max_put_oi"].replace({0: np.nan})

    # Local‐percent: what fraction of this expiry’s total is the spike?
    df["max_call_local_pct"]  = df["max_vol_call"]  / df["call_vol_sum"].replace({0: np.nan})
    df["max_put_local_pct"]   = df["max_vol_put"]   / df["put_vol_sum"].replace({0: np.nan})

    # Global‐percent: what fraction of the entire chain’s total is the spike?
    df["max_call_global_pct"] = df["max_vol_call"]  / df["call_vol_total"].replace({0: np.nan})
    df["max_put_global_pct"]  = df["max_vol_put"]   / df["put_vol_total"].replace({0: np.nan})

    # 3) Build the universe of “unusualness” metrics
    ratio_cols = [
        # volume slots
        "max_vol_call", "second_vol_call", "third_vol_call",
        "max_vol_put",  "second_vol_put",  "third_vol_put",
        # volume→OI ratios
        "call_vol_to_oi", "put_vol_to_oi",
        # local and global percentages
        "max_call_local_pct",  "max_put_local_pct",
        "max_call_global_pct", "max_put_global_pct",
    ]

    # 4) Pick the single (row,metric) with the highest score per ticker
    top_rows = []
    for sym, grp in df.groupby("symbol", sort=False):
        sub = grp[ratio_cols].dropna(how="all")  # only rows with at least one non-NaN
        if sub.empty:
            continue

        stacked = sub.stack().dropna()
        (row_idx, metric) = stacked.idxmax()
        score = stacked.loc[(row_idx, metric)]

        chosen = grp.loc[row_idx].copy()
        chosen["metric"] = metric
        chosen["score"]  = score
        top_rows.append(chosen)

    report = pd.DataFrame(top_rows).sort_values("score", ascending=False)

    # 5) Print nicely
    print(f"{'TICKER':<6}  {'EXPIRY':<10}  {'METRIC':<20}  {'SCORE':>10}")
    print("-"*50)
    for _, r in report.iterrows():
        exp = r.expiration_date.strftime("%Y-%m-%d")
        sc  = f"{r.score:.2f}"
        print(f"{r.symbol:<6}  {exp:<10}  {r.metric:<20}  {sc:>10}")

if __name__ == "__main__":
    unusual_volume_report()
