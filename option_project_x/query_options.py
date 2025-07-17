# File: query_options.py
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import json
import logging
import time
import os
                        # // HELPER FUNCTIONS:

# Source of Truth JSON File -- tickers.json:
tickers_file = 'tickers.json'

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("ingest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Divide-against-zero-helper
def safe_divide(n, d):
    if d and not np.isnan(d):
        return n / d
    return 0.0


# 'Unusualness' helper for giving ranks+labels to individual unusual volume ratios for call/put contracts
def interpret_unusualness(ratio):
    if np.isnan(ratio) or ratio == 0:
        return "Unranked"
    if np.isinf(ratio):
        return "Inf Rank"

    if ratio <= 2.0:
        return "unranked"
    elif ratio <= 3.0:
        return "unranked"
    elif ratio <= 5.0:
        return "3-5x"
    elif ratio <= 7.0:
        return "5-7x"
    elif ratio <= 10.0:
        return "7-10x"
    elif ratio <= 20.0:
        return "10-20x"
    elif ratio <= 40.0:
        return "20-40x"
    elif ratio <= 60.0:
        return "40-60x"
    else:
        return "60x"


                    # // Main Option Ingest (Builds a FINAL DICTIONARY of option data, and the UNUSUAL VOLUME report)


# 'UNUSUAL_THRESHOLDS' & 'OTM_THRESHOLDS' are both used to build reports and tables for SQLite DB downstream
# - used  for 'unusual volume report'
UNUSUAL_THRESHOLDS = {"5-7x", "7-10x", "10-20x", "20-40x", "40-60x️", "60x"}

# - used for 'market structure report'
OTM_THRESHOLDS = 0.30   # -- ±30% OTM


DB_FILE = os.path.abspath("options.db")
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)


                        # // MAIN INGEST
def pull_option_chain(ticker, expiration_date):
    # building blocks for the call and put option chains
    try:
        option_chain = ticker.option_chain(expiration_date)
    except Exception as e:
        logger.warning(f"No option chain for {ticker.ticker} @ {expiration_date}: {e}")
        return None
        # defensive check
    if option_chain is None or option_chain.calls is None or option_chain.puts is None:
        logger.warning(f"Empty option chain for {ticker.ticker} @ {expiration_date}")
        return None

    calls, puts = option_chain.calls.copy(), option_chain.puts.copy()

    # quick sanitization
    calls[['volume', 'openInterest']] = calls[['volume', 'openInterest']].fillna(0)
    puts[['volume', 'openInterest']] = puts[['volume', 'openInterest']].fillna(0)

                        # // OI‑Based Logic -->
    call_OI = calls[calls.openInterest > 0]
    put_OI = puts[puts.openInterest > 0]

            # calls
    if not call_OI.empty:
        calls_sorted_by_OI = call_OI.sort_values('openInterest', ascending=False).iloc[0]
        call_contract_with_largest_OI = (calls_sorted_by_OI.strike, calls_sorted_by_OI.volume, calls_sorted_by_OI.openInterest)
    else:
        call_contract_with_largest_OI = (None, 0.0, 0)

            # puts
    if not put_OI.empty:
        put_options_sorted_by_OI = put_OI.sort_values('openInterest', ascending=False).iloc[0]
        put_contract_with_largest_OI = (put_options_sorted_by_OI.strike, put_options_sorted_by_OI.volume, put_options_sorted_by_OI.openInterest)
    else:
        put_contract_with_largest_OI = (None, 0.0, 0)

            # SUM of option chains 'openInterest'
    call_options_OI_sum = calls.openInterest.sum()
    put_options_OI_sum = puts.openInterest.sum()


                    # // Volume‑based logic
    valid_call_vol = calls[calls.volume > 0]
    valid_put_vol = puts[puts.volume > 0]

        # calls
    if not valid_call_vol.empty:
        calls_sorted_by_volume = valid_call_vol.sort_values('volume', ascending=False).iloc[0]
        call_contract_with_largest_volume = (calls_sorted_by_volume.strike, calls_sorted_by_volume.volume, calls_sorted_by_volume.openInterest)
        call_last_price = float(calls_sorted_by_volume.lastPrice)
        call_total_spent = int(call_last_price * 100 * calls_sorted_by_volume.volume)
    else:
        call_contract_with_largest_volume = (None, 0.0, 0)
        call_last_price = 0.0
        call_total_spent = 0


        # puts
    if not valid_put_vol.empty:
        puts_sorted_by_volume = valid_put_vol.sort_values('volume', ascending=False).iloc[0]
        put_contract_with_largest_volume = (puts_sorted_by_volume.strike, puts_sorted_by_volume.volume, puts_sorted_by_volume.openInterest)
        put_last_price = float(puts_sorted_by_volume.lastPrice)
        put_total_spent = int(put_last_price * 100 * puts_sorted_by_volume.volume)
    else:
        put_contract_with_largest_volume = (None, 0.0, 0)
        put_last_price = 0.0
        put_total_spent = 0

        # SUM of option chains 'volume'
    call_options_volume_sum = calls.volume.sum()
    put_options_volume_sum = puts.volume.sum()

        # 'Unusual Volume' section
    top_call_volume_to_oi = safe_divide(call_contract_with_largest_volume[1], call_contract_with_largest_volume[2])
    top_put_volume_to_oi = safe_divide(put_contract_with_largest_volume[1], put_contract_with_largest_volume[2])

    top_call_volume_to_chainOI = safe_divide(call_contract_with_largest_volume[1], call_options_OI_sum)
    top_put_volume_to_chainOI = safe_divide(put_contract_with_largest_volume[1], put_options_OI_sum)

    unusual_call = (top_call_volume_to_oi * 0.75) + (top_call_volume_to_chainOI * 0.25)
    unusual_put = (top_put_volume_to_oi * 0.75) + (top_put_volume_to_chainOI * 0.25)

    # Small helper to convert data types to make the PostgreSQL-friendly
    def to_py_tuple(tup):
        """Convert (strike, volume, OI) into native Python types,
           treating None→0.0 for strike and None→0 for volume/OI."""
        strike, vol, oi = tup
        # if there's literally no contract, give zeros
        if strike is None:
            return (0.0, 0, 0)
        # otherwise convert normally
        return (
            float(strike),
            int(vol),
            int(oi),
        )
    # // FINAL DICTIONARY:
    return {
        "ticker": ticker.ticker,
        "expiration_date": expiration_date,
        # Open Interest
        "call_contract_with_largest_OI": to_py_tuple(call_contract_with_largest_OI),
        "put_contract_with_largest_OI": to_py_tuple(put_contract_with_largest_OI),
        "call_options_OI_sum": int(call_options_OI_sum),
        "put_options_OI_sum": int(put_options_OI_sum),
        # Volume
        "call_with_the_largest_volume": to_py_tuple(call_contract_with_largest_volume),
        "put_with_the_largest_volume": to_py_tuple(put_contract_with_largest_volume),
        "call_options_volume_sum": float(call_options_volume_sum),
        "put_options_volume_sum": float(put_options_volume_sum),
        # Unusual Report
        "call_unusualness": interpret_unusualness(unusual_call),
        "put_unusualness": interpret_unusualness(unusual_put),
        "call_last_price": call_last_price,
        "call_total_spent": call_total_spent,
        "put_last_price": put_last_price,
        "put_total_spent": put_total_spent,
    }


# ── load & flatten tickers.json ──
with open("tickers.json") as f:
    raw = json.load(f)

all_tickers = [
    {
        "symbol":    t["symbol"],
        "full_name": t["full_name"],
        "sector":    sector,
        "industry":  industry
    }
    for sector, inds in raw.items()
    for industry, ts in inds.items()
    for t in ts
]

def already_done_tickers():
    conn = sqlite3.connect("options.db")
    try:
        df = pd.read_sql_query("SELECT DISTINCT ticker FROM option_chain", conn)
        done = set(df["ticker"])
    except Exception as e:
        # if the table doesn't exist yet, treat it as “no tickers done”
        done = set()
    finally:
        conn.close()
    return done

# expose for orchestrator
options_dictionary      = []
unusual_volume_report   = []

done       = already_done_tickers()
all_symbols = {info["symbol"] for info in all_tickers}

# if we've already ingested every symbol, start fresh
if done >= all_symbols:
    logger.info("✅ All tickers already ingested—resetting to start over.")
    done.clear()

for info in all_tickers:
    sym = info["symbol"]
    if sym in done:
        logger.info(f"Skipping {sym}, already in DB")
        continue

    logger.info(f"Starting {sym}")
    ticker = yf.Ticker(sym)
    try:
        expirations = ticker.options
    except Exception as e:
        logger.error(f"Failed fetching expirations for {sym}: {e}", exc_info=True)
        time.sleep(1.1)
        continue

    rows_this_ticker    = []
    unusual_this_ticker = []

    # 1) pull every expiration
    for exp in expirations:
        row = pull_option_chain(ticker, exp)
        if not row:
            continue

        # inject metadata
        row.update({
            "ticker":   sym,
            "sector":   info["sector"],
            "industry": info["industry"]
        })
        rows_this_ticker.append(row)

        # 2) build unusual‑volume rows
        if row["call_unusualness"] in UNUSUAL_THRESHOLDS:
            unusual_this_ticker.append({
                "ticker":           sym,
                **{k: row[k] for k in ("sector","industry")},
                "expiration_date":  exp,
                "side":             "call",
                **dict(zip(
                    ["strike","volume","openInterest"],
                    row["call_with_the_largest_volume"]
                )),
                "unusualness":      row["call_unusualness"],
                "total_spent": row["call_total_spent"],
            })
        if row["put_unusualness"] in UNUSUAL_THRESHOLDS:
            unusual_this_ticker.append({
                "ticker":           sym,
                **{k: row[k] for k in ("sector","industry")},
                "expiration_date":  exp,
                "side":             "put",
                **dict(zip(
                    ["strike","volume","openInterest"],
                    row["put_with_the_largest_volume"]
                )),
                "unusualness":      row["put_unusualness"],
                "total_spent": row["put_total_spent"],
            })

    # 3) keep for in‑memory lists (if you ever want them later)
    options_dictionary.extend(rows_this_ticker)
    unusual_volume_report.extend(unusual_this_ticker)

    # 4) write as you go (survives crashes mid‑run)
    if rows_this_ticker:
        from upsert_options import upsert_rows
        upsert_rows(rows_this_ticker)
    if unusual_this_ticker:
        from upsert_options import upsert_unusual_report
        upsert_unusual_report(unusual_this_ticker)

    # ─── compute & upsert Market Structure for this ticker ───
    import sqlite3
    import pandas as pd
    from upsert_options import upsert_market_structure

    conn = sqlite3.connect("options.db")
    df_ms = pd.read_sql_query(
        """
        SELECT
          expiration_date,
          call_strike_OI   AS max_call_strike,
          put_strike_OI    AS max_put_strike,
          call_OI_OI       AS max_call_oi,
          put_OI_OI        AS max_put_oi,
          sector, industry
        FROM option_chain
        WHERE ticker = :sym
        """,
        conn,
        params={"sym": sym},
        parse_dates=["expiration_date"]
    )
    conn.close()

    if not df_ms.empty:
        # strip tz, normalize to midnight
        df_ms["expiration_date"] = (
            df_ms["expiration_date"]
            .dt.tz_localize(None)
            .dt.normalize()
        )
        today = pd.Timestamp.utcnow().normalize().date()

        # days‑to‑expiry
        df_ms["days_to_expiry"] = df_ms["expiration_date"].dt.date.map(
            lambda exp: (exp - today).days
        )
        df_ms = df_ms[df_ms["days_to_expiry"] >= 0]

        # mid‑strike vs spot & pct_diff
        df_ms["mid_strike"] = (
            df_ms["max_call_strike"] + df_ms["max_put_strike"]
        ) / 2
        try:
            last_price = yf.Ticker(sym).fast_info["last_price"]
            df_ms = df_ms.dropna(subset=["mid_strike"])
            df_ms["pct_diff"] = df_ms["mid_strike"] / last_price - 1
        except Exception:
            df_ms = df_ms.iloc[0:0]

        # filter ±30%
        df_ms = df_ms[df_ms["pct_diff"].abs() >= OTM_THRESHOLDS]

        if not df_ms.empty:
            # bullish/bearish
            df_ms["structure"] = df_ms["pct_diff"].apply(
                lambda x: "BULLISH" if x > 0 else "BEARISH"
            )
            # avg OI
            df_ms["avg_oi"] = (
                df_ms["max_call_oi"] + df_ms["max_put_oi"]
            ) / 2
            # pairedness
            df_ms["pairedness"] = df_ms.apply(
                lambda r: (
                    min(r["max_call_strike"], r["max_put_strike"])
                    / max(r["max_call_strike"], r["max_put_strike"])
                ) if (r["max_call_strike"] and r["max_put_strike"]) else 0,
                axis=1
            )
            # final score
            df_ms["final_score"] = (
                df_ms["pct_diff"].abs() * df_ms["avg_oi"]
                * (1 + df_ms["days_to_expiry"] / 365)
            ) * (1 + df_ms["pairedness"])

            # pick top row
            idx = df_ms["final_score"].idxmax()
            top = df_ms.loc[[idx]].copy()
            top["pct_diff"]     = (top["pct_diff"] * 100).round(1)
            top["avg_oi"]       = top["avg_oi"].round(0).astype(int)
            top["pairedness"]   = top["pairedness"].round(2)
            top["final_score"]  = top["final_score"].round(2)
            top["expiration_date"] = top["expiration_date"].dt.strftime("%Y-%m-%d")

            ms_row = top[[
                "expiration_date","sector","industry","structure",
                "pct_diff","avg_oi","days_to_expiry",
                "pairedness","final_score"
            ]].iloc[0].to_dict()
            ms_row["ticker"] = sym

            upsert_market_structure([ms_row])

    logger.info(f"Finished {sym}: {len(rows_this_ticker)} rows / {len(unusual_this_ticker)} unusual")
    time.sleep(1.1)