
import yfinance as yf

def interpret_unusualness(ratio):
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

def pull_ticker_option_chain(ticker, expiration_date):
    # Fetch ticker option chain -> naturally in dataframe format
    option_chain = ticker.option_chain(expiration_date)
    call_options_df = option_chain.calls
    put_options_df = option_chain.puts

            # // 'Open Interest' ----
    # *SORT options by 'openInterest' (OI):
    call_options_sorted_by_OI = call_options_df.sort_values(by='openInterest', ascending=False)
    put_options_sorted_by_OI = put_options_df.sort_values(by='openInterest', ascending=False)

    # *FIND contracts /w the largest 'openInterest' (OI):
    call_contract_with_largest_OI = tuple(call_options_sorted_by_OI.iloc[0][['strike', 'volume', 'openInterest']])
    put_contract_with_largest_OI = tuple(put_options_sorted_by_OI.iloc[0][['strike', 'volume', 'openInterest']])

    # *TOTAL 'openInterest' (OI) for given expiration_date:
    call_options_OI_sum = call_options_df['openInterest'].sum()
    put_options_OI_sum = put_options_df['openInterest'].sum()

            # // 'Volume' ----
    # *SORT options by 'volume'
    call_options_sorted_by_volume = call_options_df.sort_values(by='volume', ascending=False)
    put_options_sorted_by_volume = put_options_df.sort_values(by='volume', ascending=False)

    # *FIND call/put contracts with the largest 'volume'
    call_contract_with_largest_volume = tuple(call_options_sorted_by_volume.iloc[0][['strike', 'volume', 'openInterest']])
    put_contract_with_largest_volume = tuple(put_options_sorted_by_volume.iloc[0][['strike', 'volume', 'openInterest']])

    # *TOTAL volume for given expiration_date
    call_options_volume_sum = call_options_df['volume'].sum()
    put_options_volume_sum = put_options_df['volume'].sum()

    unusual_call_volume_to_oi = call_contract_with_largest_volume[1] / call_contract_with_largest_volume[2]
    unusual_put_volume_to_oi = put_contract_with_largest_volume[1] / put_contract_with_largest_volume[2]

    # Next, we gather the ratio of the contract's volume : entire option chain's OI -- weighted, but not as heavily
    unusual_call_volume_to_chainOI = call_contract_with_largest_volume[1] / call_options_OI_sum
    unusual_put_volume_to_chainOI = put_contract_with_largest_volume[1] / put_options_OI_sum

    # Here, we weight our volume:OI much more heavily
    unusual_call_score = (unusual_call_volume_to_oi * 0.75) + (unusual_call_volume_to_chainOI * 0.25)
    unusual_put_score = (unusual_put_volume_to_oi * 0.75) + (unusual_put_volume_to_chainOI * 0.25)

    call_score_label = interpret_unusualness(unusual_call_score)
    put_score_label = interpret_unusualness(unusual_put_score)

    # This is our dictionary we can use for 'graph_builder' GUI + miscellaneous reports
    return {
        "expiration_date": expiration_date,
        # Open Interest
        "call_contract_with_largest_OI": call_contract_with_largest_OI,
        "put_contract_with_largest_OI": put_contract_with_largest_OI,
        "call_options_OI_sum": call_options_OI_sum,
        "put_options_OI_sum": put_options_OI_sum,
        # Volume
        "call_with_the_largest_volume": call_contract_with_largest_volume,
        "put_with_the_largest_volume": put_contract_with_largest_volume,
        "call_options_volume_sum": call_options_volume_sum,
        "put_options_volume_sum": put_options_volume_sum,
        # Unusual Report
        "call_volume_unusual_score": call_score_label,
        "put_volume_unusual_score": put_score_label
    }

ticker = yf.Ticker("LYFT")
results = []

for expiration_date in ticker.options:
    full_ticker_option_data = pull_ticker_option_chain(ticker, expiration_date)
    results.append(full_ticker_option_data)
    print(full_ticker_option_data)