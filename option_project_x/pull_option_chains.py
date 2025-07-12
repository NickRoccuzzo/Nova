import numpy as np
import yfinance as yf

# Define our ticker to query
ticker = yf.Ticker("WOLF")

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

            # // Sanitize the dataframes --
    calls[['volume','openInterest']] = calls[['volume','openInterest']].fillna(0)
    puts[['volume','openInterest']] = puts[['volume','openInterest']].fillna(0)

            # // OI‑based Logic
    valid_call_OI = calls[calls.openInterest > 0]
    valid_put_OI = puts[puts.openInterest > 0]

    if not valid_call_OI.empty:
        calls_sorted_by_OI = valid_call_OI.sort_values('openInterest', ascending=False).iloc[0]
        call_contract_with_largest_OI = (calls_sorted_by_OI.strike, calls_sorted_by_OI.volume, calls_sorted_by_OI.openInterest)
    else:
        call_contract_with_largest_OI = (None, 0.0, 0)

    if not valid_put_OI.empty:
        put_options_sorted_by_OI = valid_put_OI.sort_values('openInterest', ascending=False).iloc[0]
        put_contract_with_largest_OI = (put_options_sorted_by_OI.strike, put_options_sorted_by_OI.volume, put_options_sorted_by_OI.openInterest)
    else:
        put_contract_with_largest_OI = (None, 0.0, 0)

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

    call_options_volume_sum = calls.volume.sum()
    put_options_volume_sum = puts.volume.sum()

    # 4) Ratios
    top_call_volume_to_oi = safe_divide(call_contract_with_largest_volume[1], call_contract_with_largest_volume[2])
    top_put_volume_to_oi = safe_divide(put_contract_with_largest_volume[1], put_contract_with_largest_volume[2])
    top_call_volume_to_chainOI = safe_divide(call_contract_with_largest_volume[1], call_options_OI_sum)
    top_put_volume_to_chainOI = safe_divide(put_contract_with_largest_volume[1], put_options_OI_sum)

    u_call = (top_call_volume_to_oi * 0.75) + (top_call_volume_to_chainOI * 0.25)
    u_put = (top_put_volume_to_oi * 0.75) + (top_put_volume_to_chainOI * 0.25)


    return {
        "expiration_date": expiration_date,                                       # x-axis on graph_builder
        # Open Interest
        "call_contract_with_largest_OI": call_contract_with_largest_OI,           # y-axis1 -- line graph w/ markers
        "put_contract_with_largest_OI": put_contract_with_largest_OI,             # y-axis1 -- line graph w/ markers
        "call_options_OI_sum": call_options_OI_sum,                               # y-axis2 -- green bar graph
        "put_options_OI_sum": put_options_OI_sum,                                 # y-axis2 -- red bar graph
        # Volume
        "call_with_the_largest_volume": call_contract_with_largest_volume,        # not graphed
        "put_with_the_largest_volume": put_contract_with_largest_volume,          # not graphed
        "call_options_volume_sum": call_options_volume_sum,                       # not graphed
        "put_options_volume_sum": put_options_volume_sum,                         # not graphed
        # Unusual Report
        "call_unusualness": interpret_unusualness(u_call),                        # not graphed -- used for unusual_volume_report
        "put_unusualness": interpret_unusualness(u_put)                           # not graphed -- used for unusual_volume_report
    }
options_dictionary = []

for expiration_date in ticker.options:
    full_ticker_option_chain = pull_option_chain(ticker, expiration_date)
    options_dictionary.append(full_ticker_option_chain) # < FULL dataset that allows for downstream reports
    print(full_ticker_option_chain)
