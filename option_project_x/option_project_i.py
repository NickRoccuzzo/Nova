# ============================ #
# pull option chain data
# ============================ #

import yfinance as yf

# Define the {ticker}
ticker = yf.Ticker("MSFT")

# "[0]" represents the first available 'expiration_date' for {ticker}
call_options_df = ticker.option_chain(ticker.options[0]).calls
put_options_df = ticker.option_chain(ticker.options[0]).puts

# //// Helper Functions:

## Open Interest (OI) tools ##

# SORT option chains by 'openInterest' --> (quick access to the largest values)
call_options_sorted_by_OI = call_options_df.sort_values(by='openInterest', ascending=False)
put_options_sorted_by_OI = put_options_df.sort_values(by='openInterest', ascending=False)

# FIND options with the largest 'openInterest'
call_with_largest_OI = tuple(call_options_sorted_by_OI.iloc[0][['strike', 'openInterest', 'volume']])
put_with_largest_OI = tuple(put_options_sorted_by_OI.iloc[0][['strike', 'openInterest', 'volume']])
# If you need access to the raw dataframes again, uncomment out the ones below:
# call_with_largest_OI = (call_options_sorted_by_OI.iloc[0])
# put_with_largest_OI = (put_options_sorted_by_OI.iloc[0])

# TOTAL openInterest for given expiration_date
call_options_OI_sum = call_options_df['openInterest'].sum()
put_options_OI_sum = put_options_df['openInterest'].sum()

# We'll utilize these to variables to build our graphs moving forward:
#   per expiration_date, we'll graph:
# call_with_largest_OI
# put_with_largest_OI
# call_options_OI_sum
# put_options_OI_sum



## Volume ##

# SORT the Call/Put option chains by 'volume'
call_options_sorted_by_volume = call_options_df.sort_values(by='volume', ascending=False)
call_with_largest_volume = tuple(call_options_sorted_by_volume.iloc[0][['strike', 'volume']])
call_with_2nd_largest_volume = tuple(call_options_sorted_by_volume.iloc[1][['strike', 'volume']])
call_with_3rd_largest_volume = tuple(call_options_sorted_by_volume.iloc[2][['strike', 'volume']])

put_options_sorted_by_volume = put_options_df.sort_values(by='volume', ascending=False)
put_with_largest_volume = put_options_sorted_by_volume.iloc[0]
put_with_2nd_largest_volume = put_options_sorted_by_volume.iloc[1]
put_with_3rd_largest_volume = put_options_sorted_by_volume.iloc[2]

print(call_with_largest_OI)
print(put_with_largest_OI)
print(call_with_largest_volume)
print(put_with_largest_volume)
print()