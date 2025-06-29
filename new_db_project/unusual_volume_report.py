import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://option_user:option_pass@localhost:5432/tickers"

def unusual_volume_report():
    engine = create_engine(DATABASE_URL)

    # ── 1) Load metrics + global totals ────────────────────────────────────
    df = pd.read_sql(text("""
      SELECT
        om.ticker_id,
        t.symbol,
        e.expiration_date,

        om.call_vol_sum,    om.max_vol_call,    om.max_call_oi,    om.max_call_strike,
        om.second_vol_call, om.second_call_oi,  om.second_call_strike,
        om.third_vol_call,  om.third_call_oi,   om.third_call_strike,

        om.put_vol_sum,     om.max_vol_put,     om.max_put_oi,     om.max_put_strike,
        om.second_vol_put,  om.second_put_oi,   om.second_put_strike,
        om.third_vol_put,   om.third_put_oi,    om.third_put_strike,

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

    # ── 2) Drop past expirations ───────────────────────────────────────────
    today = pd.Timestamp(datetime.now().date())
    df    = df[df.expiration_date >= today]
    if df.empty:
        print("No future expirations in the data.")
        return

    # ── 3) Compute extra lenses ────────────────────────────────────────────
    df["call_vol_to_oi"]     = df.max_vol_call / df.max_call_oi.replace({0: np.nan})
    df["put_vol_to_oi"]      = df.max_vol_put  / df.max_put_oi.replace({0: np.nan})
    df["max_call_local_pct"] = df.max_vol_call / df.call_vol_sum.replace({0: np.nan})
    df["max_put_local_pct"]  = df.max_vol_put  / df.put_vol_sum.replace({0: np.nan})
    df["max_call_global_pct"]= df.max_vol_call / df.call_vol_total.replace({0: np.nan})
    df["max_put_global_pct"] = df.max_vol_put  / df.put_vol_total.replace({0: np.nan})

    # ── 4) Pick the single best metric per symbol ───────────────────────────
    ratio_cols = [
      "max_vol_call","second_vol_call","third_vol_call",
      "max_vol_put", "second_vol_put", "third_vol_put",
      "call_vol_to_oi","put_vol_to_oi",
      "max_call_local_pct","max_put_local_pct",
      "max_call_global_pct","max_put_global_pct"
    ]

    top = []
    for sym, grp in df.groupby("symbol", sort=False):
        sub = grp[ratio_cols].dropna(how="all")
        if sub.empty:
            continue
        st            = sub.stack().dropna()
        (rid, metric) = st.idxmax()
        raw_score     = st.loc[(rid, metric)]
        row           = grp.loc[rid].copy()
        row["metric_col"] = metric
        row["raw_score"]  = raw_score
        top.append(row)

    report = pd.DataFrame(top)
    if report.empty:
        print("No unusual rows found.")
        return

    # ── 5) Bring in each ticker's latest close price ────────────────────────
    price_df = pd.read_sql(text("""
      SELECT DISTINCT ON (symbol)
        symbol, close AS last_price
      FROM price_history
      ORDER BY symbol, date DESC
    """), engine)
    report = report.merge(price_df, on="symbol", how="left")

    # ── 6) Compute days_to_expiry & moneyness ──────────────────────────────
    report["days_to_expiry"] = (report.expiration_date - today).dt.days
    is_call = report.metric_col.str.endswith("call")
    diff    = np.where(
      is_call,
      report.max_call_strike - report.last_price,
      report.last_price - report.max_put_strike
    )
    report["moneyness"] = (diff / report.last_price).clip(lower=0)

    # ── 7) Build adjusted score ────────────────────────────────────────────
    report["adj_score"] = (
       report["raw_score"]
       * (1 + report["days_to_expiry"] / 365)
       * (1 + report["moneyness"])
    )

    # ── 8) Sort by adjusted score & print ──────────────────────────────────
    report = report.sort_values("adj_score", ascending=False)
    print(f"{'TICKER':<6}  {'EXPIRY':<10}  {'METRIC':<12}  {'ADJ_SCORE':>10}")
    print("-"*46)
    for _, r in report.iterrows():
        base          = r["metric_col"].replace("_vol_", "_")  # e.g. "max_call"
        strike_col    = f"{base}_strike"
        strike        = r[strike_col]
        side          = base.split("_")[-1].upper()            # CALL or PUT
        metric_display= f"${strike:.2f} {side}"
        exp           = r["expiration_date"].strftime("%Y-%m-%d")
        print(f"{r['symbol']:<6}  {exp:<10}  {metric_display:<12}  {r['adj_score']:10.2f}")

    # ── 9) Persist to feedback_table ────────────────────────────────────────
    run_date = today
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feedback_table (
          ticker       TEXT    NOT NULL,
          run_date     DATE    NOT NULL,
          expiry_date  DATE    NOT NULL,
          metric       TEXT    NOT NULL,
          closing_price NUMERIC,
          pct_change   DOUBLE PRECISION,
          volume       BIGINT,
          adj_score    DOUBLE PRECISION,
          PRIMARY KEY (ticker, run_date, expiry_date, metric)
        );
        """))

        insert_sql = text("""
        INSERT INTO feedback_table (
          ticker, run_date, expiry_date, metric,
          closing_price, pct_change, volume, adj_score
        ) VALUES (
          :ticker, :run_date, :expiry_date, :metric,
          :closing_price, :pct_change, :volume, :adj_score
        )
        ON CONFLICT (ticker, run_date, expiry_date, metric) DO UPDATE
          SET
            closing_price = COALESCE(feedback_table.closing_price, EXCLUDED.closing_price),
            pct_change    = COALESCE(feedback_table.pct_change,    EXCLUDED.pct_change),
            volume        = COALESCE(feedback_table.volume,        EXCLUDED.volume),
            adj_score     = EXCLUDED.adj_score;
        """)

        for _, r in report.iterrows():
            base   = r["metric_col"].replace("_vol_", "_")
            strike = r[f"{base}_strike"]
            side   = base.split("_")[-1].upper()
            disp   = f"${strike:.2f} {side}"

            # pull today's close/volume/%chg if present in price_history for run_date
            ph = conn.execute(text("""
              SELECT close, 
                     ((close - open)/open) AS pct_change, 
                     volume
                FROM price_history
               WHERE symbol = :sym
                 AND date   = :d
            """), {"sym": r["symbol"], "d": run_date.date()}).fetchone()

            conn.execute(insert_sql, {
                "ticker":        r["symbol"],
                "run_date":      run_date.date(),
                "expiry_date":   r["expiration_date"].date(),
                "metric":        disp,
                "closing_price": None if ph is None else float(ph.close),
                "pct_change":    None if ph is None else float(ph.pct_change),
                "volume":        None if ph is None else int(ph.volume),
                "adj_score":     float(r["adj_score"]),
            })

    print("\n✅ feedback_table updated with this run’s picks.")

if __name__ == "__main__":
    unusual_volume_report()
