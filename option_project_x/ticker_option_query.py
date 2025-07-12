# -- We would run this analysis across each [0] index available

# OPTION CHAIN Data Tool
import yfinance as yf

# Define the {ticker}
ticker = yf.Ticker("LYFT")



# "[0]" represents the first available 'expiration_date' for {ticker}
call_options_df = ticker.option_chain(ticker.options[2]).calls
put_options_df = ticker.option_chain(ticker.options[2]).puts

#                   // Open Interest (OI):

# *SORT options by 'openInterest' (OI):
call_options_sorted_by_OI = call_options_df.sort_values(by='openInterest', ascending=False)
put_options_sorted_by_OI = put_options_df.sort_values(by='openInterest', ascending=False)

# *FIND contracts /w the largest 'openInterest' (OI):
call_contract_with_largest_OI = tuple(call_options_sorted_by_OI.iloc[0][['strike', 'openInterest', 'volume']])
put_contract_with_largest_OI = tuple(put_options_sorted_by_OI.iloc[0][['strike', 'openInterest', 'volume']])
# Raw dataframes below without tuple defined:
# call_contract_with_largest_OI = (call_options_sorted_by_OI.iloc[0])
# put_contract_with_largest_OI = (put_options_sorted_by_OI.iloc[0])

# *TOTAL 'openInterest' (OI) for given expiration_date:
call_options_OI_sum = call_options_df['openInterest'].sum()
put_options_OI_sum = put_options_df['openInterest'].sum()

#       These will be the primary variables used for building 'graph_builder' (-- Plotly GUI):

#   for each [n] expiration_date,   -- (x-axis) -- anchor for the set of associated data points being
#   call_contract_with_largest_OI   -- (y-axis1) -- line graph with square marker
#   put_contract_with_largest_OI    -- (y-axis1) -- line graph with square marker
#   call_options_OI_sum             -- (y-axis2) -- bar graph with green shade
#   put_options_OI_sum              -- (y-axis2) -- bar graph with red shade



#                       // Volume Tools:
# *SORT option chains by 'volume' --> quick access to the largest contracts
call_options_sorted_by_volume = call_options_df.sort_values(by='volume', ascending=False)
put_options_sorted_by_volume = put_options_df.sort_values(by='volume', ascending=False)

# *FIND call/put contracts with the largest 'volume'
call_with_largest_volume = tuple(call_options_sorted_by_volume.iloc[0][['strike', 'volume', 'openInterest']])
put_with_largest_volume = tuple(put_options_sorted_by_volume.iloc[0][['strike', 'volume', 'openInterest']])

# *TOTAL volume for given expiration_date
call_options_volume_sum = call_options_df['volume'].sum()
put_options_volume_sum = put_options_df['volume'].sum()

#   !(*) Unusual Score (*)!

# First we gather the ratio of the contract's Volume : OI -- weighted most heavily
unusual_call_volume_to_oi = call_with_largest_volume[1] / call_with_largest_volume[2]
unusual_put_volume_to_oi = put_with_largest_volume[1] / put_with_largest_volume[2]

# Next, we gather the ratio of the contract's volume : entire option chain's OI -- weighted, but not as heavily
unusual_call_volume_to_chainOI = call_with_largest_volume[1] / call_options_OI_sum
unusual_put_volume_to_chainOI = put_with_largest_volume[1] / put_options_OI_sum

def interpret_unusualness(ratio):
    if ratio <= 1.0:
        return "Not unusual"
    elif ratio <= 1.5:
        return "Mildly Unusual"
    elif ratio <= 2.0:
        return "Unusual"
    elif ratio <= 3.0:
        return "Very Unusual"
    else:
        return "🔥 Highly Unusual"

# Here, we weight our volume:OI much more heavily
unusual_call_score = (unusual_call_volume_to_oi * 0.75) + (unusual_call_volume_to_chainOI * 0.25)
unusual_put_score = (unusual_put_volume_to_oi * 0.75) + (unusual_put_volume_to_chainOI * 0.25)

call_score_label = interpret_unusualness(unusual_call_score)
put_score_label = interpret_unusualness(unusual_put_score)

print(put_score_label)