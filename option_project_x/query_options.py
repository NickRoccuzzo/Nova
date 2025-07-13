import numpy as np
import yfinance as yf
import json
import logging
import time

# Setup logging to monitor progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("ingest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# // Basic Helpers
def safe_divide(n, d):
    if d and not np.isnan(d):
        return n / d
    return 0.0


def interpret_unusualness(ratio):
    if np.isnan(ratio) or ratio == 0:
        return "Not Unusual"
    if np.isinf(ratio):
        return "🌀 Infinite Whale 🌀"

    if ratio <= 1.0:
        return "Not Unusual"
    elif ratio <= 1.5:
        return "Mildly Unusual"
    elif ratio <= 2.0:
        return "Unusual"
    elif ratio <= 3.0:
        return "Very Unusual"
    elif ratio <= 4.0:
        return "🔥 Highly Unusual"
    elif ratio <= 5.0:
        return "🔥 Extremely Unusual 🔥"
    elif ratio <= 8.0:
        return "🌊 WHALE ACTIVITY 🌊"
    elif ratio <= 12.0:
        return "🚨 MEGA Whale Detected 🚨"
    elif ratio <= 20.0:
        return "🌋🧨 Institutional Order Flow 🧨🌋"
    else:
        return "💀 ABSURD ORDER SIZE 💀"


# Main function
def pull_option_chain(ticker, expiration_date):
    option_chain = ticker.option_chain(expiration_date)
    calls, puts = option_chain.calls.copy(), option_chain.puts.copy()
            # // Sanitize the dataframes
    calls[['volume','openInterest']] = calls[['volume','openInterest']].fillna(0)
    puts[['volume','openInterest']] = puts[['volume','openInterest']].fillna(0)
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

    if not valid_call_vol.empty:
        calls_sorted_by_volume = valid_call_vol.sort_values('volume', ascending=False).iloc[0]
        call_contract_with_largest_volume = (calls_sorted_by_volume.strike, calls_sorted_by_volume.volume, calls_sorted_by_volume.openInterest)
    else:
        call_contract_with_largest_volume = (None, 0.0, 0)

    if not valid_put_vol.empty:
        puts_sorted_by_volume = valid_put_vol.sort_values('volume', ascending=False).iloc[0]
        put_contract_with_largest_volume = (puts_sorted_by_volume.strike, puts_sorted_by_volume.volume, puts_sorted_by_volume.openInterest)
    else:
        put_contract_with_largest_volume = (None, 0.0, 0)
    # SUM of option chains 'volume'
    call_options_volume_sum = calls.volume.sum()
    put_options_volume_sum = puts.volume.sum()

            # // Unusual Volume Builders (used for reporting/etc)
    top_call_volume_to_oi = safe_divide(call_contract_with_largest_volume[1], call_contract_with_largest_volume[2])
    top_put_volume_to_oi = safe_divide(put_contract_with_largest_volume[1], put_contract_with_largest_volume[2])

    top_call_volume_to_chainOI = safe_divide(call_contract_with_largest_volume[1], call_options_OI_sum)
    top_put_volume_to_chainOI = safe_divide(put_contract_with_largest_volume[1], put_options_OI_sum)

    unusual_call = (top_call_volume_to_oi * 0.75) + (top_call_volume_to_chainOI * 0.25)
    unusual_put = (top_put_volume_to_oi * 0.75) + (top_put_volume_to_chainOI * 0.25)

    # small helper to convert data types to make the PostgreSQL-friendly
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
    # // DICTIONARY:
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

# 3) Load & flatten tickers.json
with open("tickers.json") as f:
    raw = json.load(f)


all_tickers = []
for sector, industries in raw.items():
    for industry, tickers in industries.items():
        for t in tickers:
            all_tickers.append({
                "symbol":     t["symbol"],
                "full_name":  t["full_name"],
                "sector":     sector,
                "industry":   industry
            })

# 4) Build both lists
options_dictionary = []
unusual_volume_report = []

for info in all_tickers:
    sym = info["symbol"]
    sector = info["sector"]
    industry = info["industry"]

    logger.info(f"Starting ingestion for {sym}")
    ticker = yf.Ticker(sym)

    try:
        expirations = ticker.options
    except Exception as e:
        logger.error(f"Failed to fetch expirations for {sym}: {e}", exc_info=True)
        # even on error, pause before next ticker
        time.sleep(1.1)
        continue

    for exp in ticker.options:
        row = pull_option_chain(ticker, exp)
        # inject metadata
        row["ticker"] = sym
        row["sector"] = sector
        row["industry"] = industry

        options_dictionary.append(row)
        logger.info(f"Finished ingestion for {sym} (total rows: {len(options_dictionary)})")

        # Unusual Thresholds for building our report
        UNUSUAL_THRESHOLDS = {"Unusual", "Very Unusual", "🔥 Highly Unusual", "🔥 Extremely Unusual 🔥", "🌊 WHALE ACTIVITY 🌊", "🚨 MEGA Whale Detected 🚨", "🌋🧨 Institutional Order Flow 🧨🌋", "💀 ABSURD ORDER SIZE 💀"}

        # threshold‐filter
        if row["call_unusualness"] in UNUSUAL_THRESHOLDS:
            unusual_volume_report.append({
                "ticker":          sym,
                "sector":          sector,
                "industry":        industry,
                "expiration_date": row["expiration_date"],
                "side":            "call",
                "strike":          row["call_with_the_largest_volume"][0],
                "volume":          row["call_with_the_largest_volume"][1],
                "openInterest":    row["call_with_the_largest_volume"][2],
                "unusualness":     row["call_unusualness"]
            })

        if row["put_unusualness"] in UNUSUAL_THRESHOLDS:
            unusual_volume_report.append({
                "ticker":          sym,
                "sector":          sector,
                "industry":        industry,
                "expiration_date": row["expiration_date"],
                "side":            "put",
                "strike":          row["put_with_the_largest_volume"][0],
                "volume":          row["put_with_the_largest_volume"][1],
                "openInterest":    row["put_with_the_largest_volume"][2],
                "unusualness":     row["put_unusualness"]
            })
    time.sleep(1.1)
