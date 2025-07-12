# Option Chain Data Tool
import yfinance as yf

# Define the {ticker}
ticker = yf.Ticker("MSTR")

# "[0]" represents the first available 'expiration_date' for {ticker}
call_options_df = ticker.option_chain(ticker.options[0]).calls
put_options_df = ticker.option_chain(ticker.options[0]).puts


# //// Helper Functions:
    # We'll have Open Interest tooling for graph building, and Volume tooling for reports + miscellaneous usage

#       ____ Open Interest (OI) TOOLING ____ ///

# *SORT options by 'openInterest' (OI)
call_options_sorted_by_OI = call_options_df.sort_values(by='openInterest', ascending=False)
put_options_sorted_by_OI = put_options_df.sort_values(by='openInterest', ascending=False)

# *FIND contracts /w the largest 'openInterest' (OI)
call_contract_with_largest_OI = tuple(call_options_sorted_by_OI.iloc[0][['strike', 'openInterest', 'volume']])
put_contract_with_largest_OI = tuple(put_options_sorted_by_OI.iloc[0][['strike', 'openInterest', 'volume']])
# Raw dataframes below without tuple:
# call_contract_with_largest_OI = (call_options_sorted_by_OI.iloc[0])
# put_contract_with_largest_OI = (put_options_sorted_by_OI.iloc[0])

# *TOTAL 'openInterest' (OI) for given expiration_date
call_options_OI_sum = call_options_df['openInterest'].sum()
put_options_OI_sum = put_options_df['openInterest'].sum()

# We'll utilize these variables in 'graph_builder' now:
#   - per expiration_date, we'll graph:
#   call_contract_with_largest_OI
#   put_contract_with_largest_OI
#   call_options_OI_sum
#   put_options_OI_sum



#        ____ VOLUME TOOLING ____ ///

# *SORT option chains by 'volume' --> quick access to the largest contracts
call_options_sorted_by_volume = call_options_df.sort_values(by='volume', ascending=False)
put_options_sorted_by_volume = put_options_df.sort_values(by='volume', ascending=False)

# *FIND options with the largest 'volume'
call_with_largest_volume = tuple(call_options_sorted_by_volume.iloc[0][['strike', 'volume', 'openInterest']])
put_with_largest_volume = tuple(put_options_sorted_by_volume.iloc[0][['strike', 'volume', 'openInterest']])

# *TOTAL volume for given expiration_date
call_options_volume_sum = call_options_df['volume'].sum()
put_options_volume_sum = put_options_df['volume'].sum()

# /// Unusual Score

# First we gather the ratio of the contract's volume : OI
unusual_call_volume_to_oi = call_with_largest_volume[1] / call_with_largest_volume[2]
unusual_put_volume_to_oi = put_with_largest_volume[1] / put_with_largest_volume[2]

# Next, we gather the ratio of the conract's volume : entire option chain's OI
unusual_call_volume_to_chainOI = call_with_largest_volume[1] / call_options_OI_sum
unusual_put_volume_to_chainOI = put_with_largest_volume[1] / put_options_OI_sum

def interpret_unusualness(ratio):
    if ratio <= 1.0:
        return "Not unusual"
    elif ratio <= 1.5:
        return "Mildly unusual"
    elif ratio <= 2.0:
        return "Unusual"
    elif ratio <= 3.0:
        return "Very Unusual"
    else:
        return "🔥 Highly Unusual"

# Volume spikes are more important, so we weight them more heavily
unusual_call_score = (unusual_call_volume_to_oi * 0.7) + (unusual_call_volume_to_chainOI * 0.3)
unusual_put_score = (unusual_put_volume_to_oi * 0.7) + (unusual_put_volume_to_chainOI * 0.3)

call_score_label = interpret_unusualness(unusual_call_score)
put_score_label = interpret_unusualness(unusual_put_score)

print(f"🔔 Calls Combined Score = {unusual_call_score:.2f} → {call_score_label}")
print(f"🔔 Puts Combined Score = {unusual_put_score:.2f} → {put_score_label}")