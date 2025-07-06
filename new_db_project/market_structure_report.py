import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from db_config import POSTGRES_DB_URL

BULK_THRESHOLD = 0.30  # require at least ±30% OTM shift

def market_structure_report():
    engine = create_engine(POSTGRES_DB_URL)

    # 1) Pull strikes + open‐interest
    metrics_sql = """
    SELECT
      om.ticker_id,
      t.symbol,
      e.expiration_date,
      om.max_call_strike,
      om.max_put_strike,
      om.max_call_oi,
      om.max_put_oi
    FROM option_metrics om
    JOIN tickers     t USING (ticker_id)
    JOIN expirations e USING (expiration_id)
    """
    df = pd.read_sql(text(metrics_sql), engine, parse_dates=["expiration_date"])
    if df.empty:
        print("No option_metrics data.")
        return

    # 2) Get latest close price for each symbol
    price_sql = """
    SELECT DISTINCT ON (symbol)
      symbol,
      close AS last_price
    FROM price_history
    ORDER BY symbol, date DESC
    """
    price_df = pd.read_sql(text(price_sql), engine)
    df = df.merge(price_df, on="symbol", how="left") \
           .dropna(subset=["last_price"])
    if df.empty:
        print("No matching price_history data.")
        return

    # 3) Compute days to expiry & filter out past expiries
    today = pd.Timestamp(datetime.now().date())
    df["days_to_expiry"] = (df["expiration_date"] - today).dt.days
    df = df[df["days_to_expiry"] >= 0]
    if df.empty:
        print("No future expirations.")
        return

    # 4) Compute mid‐chain strike and % difference from spot
    df["mid_strike"] = (df["max_call_strike"] + df["max_put_strike"]) / 2
    df["pct_diff"]   = df["mid_strike"] / df["last_price"] - 1

    # 5) Keep only those with |pct_diff| ≥ threshold
    df = df[df["pct_diff"].abs() >= BULK_THRESHOLD]
    if df.empty:
        print(f"No expiries outside ±{int(BULK_THRESHOLD*100)}% of spot.")
        return

    # 6) Classify bullish vs bearish
    df["structure"] = df["pct_diff"].apply(lambda x: "BULLISH" if x > 0 else "BEARISH")

    # 7) Weight by open‐interest
    df["avg_oi"] = (df["max_call_oi"] + df["max_put_oi"]) / 2

    # 8) Base adjusted score: |pct_diff| × avg_oi × (1 + days_to_expiry/365)
    df["adj_score"] = (
        df["pct_diff"].abs()
        * df["avg_oi"]
        * (1 + df["days_to_expiry"] / 365)
    )

    # 9) Pairedness boost: how close together the two strikes sit
    df["pairedness"] = df.apply(
        lambda r: (
            min(r["max_call_strike"], r["max_put_strike"])
            / max(r["max_call_strike"], r["max_put_strike"])
        ) if (r["max_call_strike"] and r["max_put_strike"]) else 0,
        axis=1
    )

    # 10) Final score = adj_score × (1 + pairedness)
    df["final_score"] = df["adj_score"] * (1 + df["pairedness"])

    # 11) Drop any rows without a valid final_score
    df = df[df["final_score"].notna()]
    if df.empty:
        print("No valid structures after scoring.")
        return

    # 12) Per‐symbol pick the single row with highest final_score
    idx = (
        df.groupby("symbol")["final_score"]
          .idxmax()
          .dropna()
          .astype(int)
          .tolist()
    )
    top = df.loc[idx].sort_values("final_score", ascending=False)

    # 13) Print the report
    print(f"{'TICKER':<6}  {'EXPIRY':<10}  {'STRUCTURE':<8}  "
          f"{'%DIFF':>7}  {'AVG_OI':>8}  {'DTE':>4}  {'PAIR':>6}  {'SCORE':>12}")
    print("-" * 80)
    for _, r in top.iterrows():
        exp = r["expiration_date"].strftime("%Y-%m-%d")
        pct = r["pct_diff"] * 100
        oi = int(r["avg_oi"])
        dte = int(r["days_to_expiry"])
        pair = r["pairedness"]
        score = r["final_score"]
        print(f"{r['symbol']:<6}  {exp:<10}  {r['structure']:<8}  "
              f"{pct:7.1f}%  {oi:8d}  {dte:4d}  {pair:6.2f}  {score:12.2f}")


if __name__ == "__main__":
    market_structure_report()
