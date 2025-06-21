import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

DATABASE_URL = "postgresql://option_user:option_pass@localhost:5432/tickers"
LOOKBACK_DAYS = 90
TOP_K = 10
LAG_THRESHOLD = 0.05  # 5 percentage points

def correlations_report():
    engine = create_engine(DATABASE_URL)

    # 1) Fetch last 90 days of closing prices
    today = pd.Timestamp(datetime.now().date())
    start_date = today - pd.Timedelta(days=LOOKBACK_DAYS)
    price_df = pd.read_sql(
        text("""
            SELECT symbol, date, close
              FROM price_history
             WHERE date >= :start
             ORDER BY date ASC
        """),
        engine,
        params={"start": start_date}
    )
    if price_df.empty:
        print("No price history for the last 90 days.")
        return

    # 2) Pivot to dates×symbols
    pivot = price_df.pivot(index="date", columns="symbol", values="close")

    # 3) Drop any symbols with missing data in the window
    pivot = pivot.dropna(axis=1)
    if pivot.shape[1] < 2:
        print("Not enough symbols with complete data.")
        return

    # 4) Compute daily returns and correlation matrix
    returns = pivot.pct_change().dropna(how="all")
    corr = returns.corr()

    # 5) Compute each symbol's total % change over the period
    total_change = pivot.iloc[-1] / pivot.iloc[0] - 1

    results = []
    for sym in corr.columns:
        # 6) Top-K correlators for this symbol (exclude itself)
        peers = corr[sym].drop(labels=[sym]).nlargest(TOP_K).index.tolist()
        if not peers:
            continue

        # 7) Compare each peer's total_change vs this symbol
        sym_change = total_change[sym]
        peer_changes = total_change.loc[peers]

        # 8) Count peers that are at least LAG_THRESHOLD ahead
        ahead_peers = peer_changes[peer_changes - sym_change >= LAG_THRESHOLD]
        score = len(ahead_peers)

        results.append({
            "symbol":       sym,
            "pct_change":   sym_change,
            "score":        score,
            "ahead_peers":  ", ".join(ahead_peers.index)
        })

    if not results:
        print("No symbols found with lagging correlators.")
        return

    # 9) Build DataFrame, sort by score descending
    report_df = pd.DataFrame(results)
    report_df = report_df.sort_values("score", ascending=False)

    # 10) Print report
    print(f"{'TICKER':<6}  {'%Δ':>7}  {'SCORE':>5}  {'PEERS_AHEAD'}")
    print("-" * 60)
    for _, row in report_df.iterrows():
        sym   = row["symbol"]
        pct   = row["pct_change"] * 100
        sc    = int(row["score"])
        peers = row["ahead_peers"]
        print(f"{sym:<6}  {pct:7.1f}%  {sc:5d}   {peers}")

if __name__ == "__main__":
    correlations_report()