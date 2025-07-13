import numpy as np
import yfinance as yf


            # DEFINE TICKER(s)
ticker = yf.Ticker("AAPL")


def safe_divide(n, d):
    if d and not np.isnan(d):
        return n / d
    return 0.0


def interpret_unusualness(ratio):
    if np.isnan(ratio) or ratio == 0:
        return "Not Unusual"
    if np.isinf(ratio):
        return "🔥 Highly Unusual"
    if ratio <= 1.0:
        return "Not Unusual"
    elif ratio <= 1.5:
        return "Mildly Unusual"
    elif ratio <= 2.0:
        return "Unusual"
    elif ratio <= 3.0:
        return "Very Unusual"
    else:
        return "🔥 Highly Unusual"


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
        # tup == (strike, volume, openInterest)
        return (float(tup[0]), int(tup[1]), int(tup[2]))
    # // DICTIONARY:
    return {
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
options_dictionary = []

for expiration_date in ticker.options:
    full_ticker_option_chain = pull_option_chain(ticker, expiration_date)
    options_dictionary.append(full_ticker_option_chain) # < FULL dataset that allows for downstream reports

    # // UNUSUAL VOLUME REPORT
        # -- 'unusual' thresholds that are most important to the report
    unusual_volume_report_thresholds = {"Unusual", "Very Unusual", "🔥 Highly Unusual"}
    SYMBOL = ticker.ticker

    # Unusual Volume Report will track the 'options_dictionary' for the most unusual contracts for this ticker
    unusual_volume_report = []
    for entry in options_dictionary:
        # check call side
        if entry["call_unusualness"] in unusual_volume_report_thresholds:
            unusual_volume_report.append({
                "ticker": SYMBOL,
                "expiration_date": entry["expiration_date"],
                "side": "call",
                **dict(zip(
                    ["strike", "volume", "openInterest"],
                    entry["call_with_the_largest_volume"]
                )),
                "unusualness": entry["call_unusualness"]
            })
        # check put side
        if entry["put_unusualness"] in unusual_volume_report_thresholds:
            unusual_volume_report.append({
                "ticker": SYMBOL,
                "expiration_date": entry["expiration_date"],
                "side": "put",
                **dict(zip(
                    ["strike", "volume", "openInterest"],
                    entry["put_with_the_largest_volume"]
                )),
                "unusualness": entry["put_unusualness"]
            })

