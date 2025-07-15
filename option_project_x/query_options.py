# File: query_options.py
from upsert_options import engine, metadata
import sqlite3
import numpy as np
import yfinance as yf
import json
import logging
import time
import pandas as pd
import os

# // Setup logging to watch progress // catch any new errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("ingest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# // Divide-against-zero-helper
def safe_divide(n, d):
    if d and not np.isnan(d):
        return n / d
    return 0.0


# // 'Unusual Threshold' for labeling unusual volume ratios
def interpret_unusualness(ratio):
    if np.isnan(ratio) or ratio == 0:
        return "Not Unusual"
    if np.isinf(ratio):
        return "INF"

    if ratio <= 1.0:
        return "Not Unusual"
    elif ratio <= 1.5:
        return "Mildly Unusual"
    elif ratio <= 2.0:
        return "Unusual"
    elif ratio <= 3.0:
        return "Very Unusual"
    elif ratio <= 4.0:
        return "Highly Unusual 🔥"
    elif ratio <= 5.0:
        return "Extremely Unusual 🔥🔥"
    elif ratio <= 8.0:
        return "LARGE Whale 🌊"
    elif ratio <= 12.0:
        return "MEGA Whale 🌊🌊"
    elif ratio <= 20.0:
        return "🌊🏛️ Institutional Whale 🏛️🌊"
    else:
        return "💀🗡️ EXTREME Outlier 🗡️💀"


                    # // Main Option Ingest (Builds a FINAL DICTIONARY of option data, and the UNUSUAL VOLUME report)


# Unusual Thresholds for building our report
UNUSUAL_THRESHOLDS = {"LARGE Whale 🌊", "MEGA Whale 🌊🌊", "🌊🏛️ Institutional Whale 🏛️🌊", "💀🗡️ EXTREME Outlier 🗡️💀"}


def pull_option_chain(ticker, expiration_date):
    # building blocks for option_chain
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

                        # // OI‑based Logic
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
    else:
        call_contract_with_largest_volume = (None, 0.0, 0)

        # puts
    if not valid_put_vol.empty:
        puts_sorted_by_volume = valid_put_vol.sort_values('volume', ascending=False).iloc[0]
        put_contract_with_largest_volume = (puts_sorted_by_volume.strike, puts_sorted_by_volume.volume, puts_sorted_by_volume.openInterest)
    else:
        put_contract_with_largest_volume = (None, 0.0, 0)

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
    }


# // Load & flatten tickers.json
# ── load tickers.json ──
with open("tickers.json") as f:
    raw = json.load(f)

all_tickers = [
    {"symbol": t["symbol"], "full_name": t["full_name"],
     "sector": sector, "industry": industry}
    for sector, inds in raw.items()
    for industry, ts in inds.items()
    for t in ts
]

def already_done_tickers():
    conn = sqlite3.connect("options.db")
    df = pd.read_sql_query("SELECT DISTINCT ticker FROM option_chain", conn)
    conn.close()
    return set(df["ticker"])

# expose these for our orchestrator
options_dictionary = []
unusual_volume_report = []

done = already_done_tickers()

all_symbols = {info["symbol"] for info in all_tickers}
if done >= all_symbols:
    logger.info("✅ All tickers have already been ingested—resetting progress to start over.")
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

    rows_this_ticker = []
    unusual_this_ticker = []

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

        # check thresholds
        if row["call_unusualness"] in UNUSUAL_THRESHOLDS:
            unusual_this_ticker.append({
                "ticker": sym, **{k: row[k] for k in ("sector","industry")},
                "expiration_date": exp, "side":"call",
                **dict(zip(["strike","volume","openInterest"], row["call_with_the_largest_volume"])),
                "unusualness": row["call_unusualness"]
            })
        if row["put_unusualness"] in UNUSUAL_THRESHOLDS:
            unusual_this_ticker.append({
                "ticker": sym, **{k: row[k] for k in ("sector","industry")},
                "expiration_date": exp, "side":"put",
                **dict(zip(["strike","volume","openInterest"], row["put_with_the_largest_volume"])),
                "unusualness": row["put_unusualness"]
            })

    # append into the module‑level lists so the orchestrator can see them too:
    options_dictionary.extend(rows_this_ticker)
    unusual_volume_report.extend(unusual_this_ticker)

    # write as you go
    if rows_this_ticker:
        from upsert_options import upsert_rows
        upsert_rows(rows_this_ticker)
    if unusual_this_ticker:
        from upsert_options import upsert_unusual_report
        upsert_unusual_report(unusual_this_ticker)

    logger.info(f"Finished {sym}: {len(rows_this_ticker)} rows / {len(unusual_this_ticker)} unusual")
    time.sleep(1.1)