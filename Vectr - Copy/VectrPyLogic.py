# -- MODULES -- #
import os
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

def save_options_data(ticker):
    """
    This function is responsible for retrieving option chain data using * yfinance *
    Each ticker gets assigned its own folder, which also contains 2 subfolders ( /CALLS/ and /PUTS/ )
    Each expiration date within the given ticker(s) option chain is iterated over to extract all available contracts

    + Parameters:
    ticker (str): The stock ticker symbol for which to fetch option chain data.
    """

    # First, we use 'os.getcwd()' to retrieve the current working directory and setup the ticker folder(s)
    base_directory = os.getcwd()

    # Create a 'ticker folder' to hold the ticker(s) CALL/PUT data
    ticker_folder = os.path.join(base_directory, ticker)
    # If the folder doesn't exist
    if not os.path.exists(ticker_folder):
        # Create the ticker_folder
        os.makedirs(ticker_folder)

    # Create separate folders for CALLS and PUTS option data
    calls_folder = os.path.join(ticker_folder, "CALLS")
    puts_folder = os.path.join(ticker_folder, "PUTS")
    os.makedirs(calls_folder, exist_ok=True)
    os.makedirs(puts_folder, exist_ok=True)

    # Fetch option chain data for the ticker using yfinance
    stock = yf.Ticker(ticker)
    exp_dates = stock.options  # List of available expiration dates

    # Check if there are any options available
    if not exp_dates:
        print(f"No option chain found for the ticker {ticker}, may not exist.")
        return

    # Retry mechanism in the event connection suddenly fails // API doesn't respond correctly on first attempt
    max_retries = 3
    retry_delay = 5  # seconds

    # Iterate over each expiration date to fetch and save the option data
    for date in exp_dates:
        for attempt in range(max_retries):
            try:
                # Fetch the option chain for the given date
                opt = stock.option_chain(date)

                # Define filenames for saving the calls and puts data
                calls_filename = os.path.join(calls_folder, f"{date.replace('-', '')}CALLS.csv")
                puts_filename = os.path.join(puts_folder, f"{date.replace('-', '')}PUTS.csv")

                # Save calls and puts data to CSV files
                opt.calls.to_csv(calls_filename)
                opt.puts.to_csv(puts_filename)
                break  # If successful, exit the retry loop

            except Exception as error_message:
                print(f"Attempt {attempt + 1} of {max_retries} failed: {error_message}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)  # Wait before retrying
                else:
                    print(f"An error occurred while processing options for {date}: {error_message}")
    pass


def preprocess_dates(data_dir, file_suffix):
    """
    Preprocess and sort option chain data by expiration dates.

    Parameters:
    - data_dir (str): Directory path containing the options data CSV files.
    - file_suffix (str): 'CALLS' or 'PUTS' to handle specific file types.

    Returns:
    - dict: A dictionary with formatted dates as keys and corresponding DataFrames as values.
    """
    # 'zoneinfo' extracts the operating system's timezone (Python 3.9+ necessary for this)
    local_time = datetime.now().astimezone()  # "local" time
    local_tz = local_time.tzinfo  # local timezone object

    sorted_data = {}

    for filename in os.listdir(data_dir):
        if filename.endswith(file_suffix + ".csv"):
            # Extract the date from the filename
            date_str = filename.split(file_suffix)[0]
            try:
                # Convert to datetime and format
                expiration_date = datetime.strptime(date_str, '%Y%m%d')

                # Load the CSV and store in the dictionary
                df = pd.read_csv(os.path.join(data_dir, filename))
                if not df.empty:
                    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
                    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
                    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                    df["lastPrice"] = pd.to_numeric(df["lastPrice"], errors="coerce")

                    # Convert lastTradeDate to datetime and then to EST
                    df["lastTradeDate"] = pd.to_datetime(df["lastTradeDate"], format="%Y-%m-%d %H:%M:%S%z", utc=True, errors="coerce")
                    df["lastTradeDate_Local"] = df["lastTradeDate"].dt.tz_convert(local_tz)
                    sorted_data[expiration_date] = df

            except ValueError as e:
                print(f"Error processing {filename}: {e}")

    # Sort by the datetime key:
    sorted_pairs = sorted(sorted_data.items(), key=lambda x: x[0])

    # Convert back to strings in the final returned dict
    # e.g., {'01/24/25': df, '01/30/25': df, ...}
    return {
        key.strftime('%m/%d/%y'): value
        for key, value in sorted_pairs
    }


def format_dollar_amount(amount):
    """
    Format a given dollar amount into a human-readable string with suffixes like 'K' for thousands and 'M' for millions.
    ( Ex. $432M )
    * These will be used in the Plotly graph annotations.
    """
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.1f}B"  # Billions with one decimal place
    elif amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"  # Millions with one decimal place
    elif amount >= 1_000:
        return f"${amount / 1_000:.1f}K"  # Thousands with one decimal place
    else:
        return f"${amount:.2f}"  # Less than a thousand with two decimal places for cents
    pass


def gather_options_data(ticker):
    """
    Gathers and returns all the raw data used by the Plotly logic,
    but without creating a figure. Useful for saving to JSON/CSV.

    Returns a dictionary with keys:
      - calls_oi
      - puts_oi
      - max_strike_calls, second_max_strike_calls, third_max_strike_calls
      - max_strike_puts, second_max_strike_puts, third_max_strike_puts
      - avg_strike
      - top_volume_contracts (list of dicts)
      - current_price
      - company_name
    """

    # 1) Locate the CALLS/PUTS directories for this ticker
    base_dir = os.getcwd()
    calls_dir = os.path.join(base_dir, ticker, "CALLS")
    puts_dir = os.path.join(base_dir, ticker, "PUTS")

    # 2) Optionally preprocess the CSVs (assuming you already have a 'preprocess_dates' function)
    #    If your script is in the same file, call it directly. Otherwise import it.
    calls_data = preprocess_dates(calls_dir, "CALLS")
    puts_data = preprocess_dates(puts_dir, "PUTS")

    # 3) Initialize dictionaries for analyzing OI and strikes
    calls_oi = {date: df['openInterest'].sum() for date, df in calls_data.items()}
    puts_oi = {date: df['openInterest'].sum() for date, df in puts_data.items()}

    max_strike_calls, second_max_strike_calls, third_max_strike_calls = {}, {}, {}
    max_strike_puts, second_max_strike_puts, third_max_strike_puts = {}, {}, {}

    avg_strike = {}
    top_volume_contracts = []

    # 4) Fetch current stock data from yfinance
    stock = yf.Ticker(ticker)
    current_data = stock.history(period="1d")
    if not current_data.empty:
        current_price = current_data['Close'].iloc[-1]
    else:
        current_price = 0.0

    company_name = stock.info.get('longName', 'N/A')

    # 5) Process sorted calls data (find top strikes, highest volume calls, etc.)
    for date, df in calls_data.items():
        if not df.empty:
            # Sort by openInterest descending
            sorted_calls = df.sort_values(by='openInterest', ascending=False)

            # Safely get up to 3 top calls by OI
            if len(sorted_calls) > 0:
                max_strike_calls[date] = sorted_calls.iloc[0]['strike']
            else:
                max_strike_calls[date] = 0

            if len(sorted_calls) > 1:
                second_max_strike_calls[date] = sorted_calls.iloc[1]['strike']
            else:
                second_max_strike_calls[date] = 0

            if len(sorted_calls) > 2:
                third_max_strike_calls[date] = sorted_calls.iloc[2]['strike']
            else:
                third_max_strike_calls[date] = 0

            # Identify the call option with the highest volume
            if df['volume'].notna().any():
                highest_volume_call = df.loc[df['volume'].idxmax()]
                total_spent = highest_volume_call['volume'] * highest_volume_call['lastPrice'] * 100
                formatted_spent = format_dollar_amount(total_spent)

                # Mark if volume > OI for 'unusual'
                unusual = highest_volume_call['volume'] > highest_volume_call['openInterest']

                top_volume_contracts.append({
                    'type': 'CALL',
                    'strike': highest_volume_call['strike'],
                    'volume': highest_volume_call['volume'],
                    'openInterest': highest_volume_call['openInterest'],
                    'date': date,
                    'total_spent': formatted_spent,
                    'unusual': unusual
                })

    # 6) Process sorted puts data (find top strikes, highest volume puts, etc.)
    for date, df in puts_data.items():
        if not df.empty:
            # Sort by openInterest descending
            sorted_puts = df.sort_values(by='openInterest', ascending=False)

            # Safely get up to 3 top puts by OI
            if len(sorted_puts) > 0:
                max_strike_puts[date] = sorted_puts.iloc[0]['strike']
            else:
                max_strike_puts[date] = 0

            if len(sorted_puts) > 1:
                second_max_strike_puts[date] = sorted_puts.iloc[1]['strike']
            else:
                second_max_strike_puts[date] = 0

            if len(sorted_puts) > 2:
                third_max_strike_puts[date] = sorted_puts.iloc[2]['strike']
            else:
                third_max_strike_puts[date] = 0

            # Identify the put option with the highest volume
            if df['volume'].notna().any():
                highest_volume_put = df.loc[df['volume'].idxmax()]
                total_spent = highest_volume_put['volume'] * highest_volume_put['lastPrice'] * 100
                formatted_spent = format_dollar_amount(total_spent)

                unusual = highest_volume_put['volume'] > highest_volume_put['openInterest']

                top_volume_contracts.append({
                    'type': 'PUT',
                    'strike': highest_volume_put['strike'],
                    'volume': highest_volume_put['volume'],
                    'openInterest': highest_volume_put['openInterest'],
                    'date': date,
                    'total_spent': formatted_spent,
                    'unusual': unusual
                })

    # 7) Optionally compute an average strike for each date, if that's part of your logic
    #    (We show an example from your original code, but feel free to omit if not needed.)
    for date in max_strike_calls.keys():
        if date in max_strike_puts and (calls_oi[date] + puts_oi[date] > 0):
            total_oi = calls_oi[date] + puts_oi[date]
            weight_calls = calls_oi[date] / total_oi if total_oi else 0
            weight_puts = puts_oi[date] / total_oi if total_oi else 0
            # A simplistic version:
            avg_strike[date] = (
                    (
                            max_strike_calls[date] * weight_calls +
                            second_max_strike_calls[date] * weight_calls +
                            third_max_strike_calls[date] * weight_calls +
                            max_strike_puts[date] * weight_puts +
                            second_max_strike_puts[date] * weight_puts +
                            third_max_strike_puts[date] * weight_puts
                    ) / (3 * (weight_calls + weight_puts))
            )
        else:
            avg_strike[date] = np.nan

    # 8) Return everything as a dictionary
    return {
        "calls_oi": calls_oi,
        "puts_oi": puts_oi,
        "max_strike_calls": max_strike_calls,
        "second_max_strike_calls": second_max_strike_calls,
        "third_max_strike_calls": third_max_strike_calls,
        "max_strike_puts": max_strike_puts,
        "second_max_strike_puts": second_max_strike_puts,
        "third_max_strike_puts": third_max_strike_puts,
        "avg_strike": avg_strike,
        "top_volume_contracts": top_volume_contracts,
        "current_price": current_price,
        "company_name": company_name
    }

def calculate_and_visualize_data(ticker, width=600, height=400):
    """
    Function that analyzes option chain data and generates Plotly visualizations for the given stock ticker(s).
    """

    # Fetch current stock data using yfinance
    stock = yf.Ticker(ticker)
    current_data = stock.history(period="1d")
    current_price = current_data['Close'].iloc[-1]  # Current closing price of the stock
    price_data = stock.history(period="2d")
    company_name = stock.info.get('longName', 'N/A')  # Retrieve company name

    # Safely compute the daily change percentage if there's at least 2 rows
    if len(price_data) > 1:
        prev_close = price_data['Close'].iloc[-2]
        daily_change_dollar = current_price - prev_close
        daily_change_pct = (daily_change_dollar / prev_close) * 100
    else:
        daily_change_dollar = 0
        daily_change_pct = 0

    # Get the current working directory and construct paths for call and put data
    base_dir = os.getcwd()
    calls_dir = os.path.join(base_dir, ticker, "CALLS")
    puts_dir = os.path.join(base_dir, ticker, "PUTS")

    # Preprocess and sort calls and puts data
    calls_data = preprocess_dates(calls_dir, "CALLS")
    puts_data = preprocess_dates(puts_dir, "PUTS")

    # Initialize dictionaries for visualization
    calls_oi = {date: df['openInterest'].sum() for date, df in calls_data.items()}
    puts_oi = {date: df['openInterest'].sum() for date, df in puts_data.items()}
    max_strike_calls, max_strike_puts = {}, {}
    second_max_strike_calls, second_max_strike_puts = {}, {}
    third_max_strike_calls, third_max_strike_puts = {}, {}
    avg_strike = {}
    top_volume_contracts = []

    local_time = datetime.now().astimezone()
    today_local_date = local_time.date()

    # Process sorted calls data
    for date, df in calls_data.items():
        if not df.empty:
            # Calculate strikes with highest open interest
            sorted_calls = df.sort_values(by='openInterest', ascending=False)
            max_strike_calls[date] = sorted_calls.iloc[0]['strike'] if not sorted_calls.empty else 0
            second_max_strike_calls[date] = sorted_calls.iloc[1]['strike'] if len(sorted_calls) > 1 else 0
            third_max_strike_calls[date] = sorted_calls.iloc[2]['strike'] if len(sorted_calls) > 2 else 0

            # Identify the call option with the highest volume
            if df['volume'].notna().any():
                highest_volume_call = df.loc[df['volume'].idxmax()]

                # Check lastTradeDate_EST
                last_trade_local = highest_volume_call["lastTradeDate_Local"]
                if pd.notnull(last_trade_local):
                    if last_trade_local.date() == today_local_date:
                        # It's traded "today" in EST, so we consider it "top volume"
                        total_spent = highest_volume_call['volume'] * highest_volume_call['lastPrice'] * 100
                        formatted_total_spent = format_dollar_amount(total_spent)
                        unusual = highest_volume_call['volume'] > highest_volume_call['openInterest']

                        top_volume_contracts.append({
                            'type': 'CALL',
                            'strike': highest_volume_call['strike'],
                            'volume': highest_volume_call['volume'],
                            'openInterest': highest_volume_call['openInterest'],
                            'date': date,
                            'total_spent': formatted_total_spent,
                            'unusual': unusual
                        })

    # Process sorted puts data
    for date, df in puts_data.items():
        if not df.empty:
            # Calculate strikes with the highest open interest
            sorted_puts = df.sort_values(by='openInterest', ascending=False)
            max_strike_puts[date] = sorted_puts.iloc[0]['strike'] if not sorted_puts.empty else 0
            second_max_strike_puts[date] = sorted_puts.iloc[1]['strike'] if len(sorted_puts) > 1 else 0
            third_max_strike_puts[date] = sorted_puts.iloc[2]['strike'] if len(sorted_puts) > 2 else 0

            # Identify the put option with the highest volume
            if df['volume'].notna().any():
                highest_volume_put = df.loc[df['volume'].idxmax()]

                last_trade_local = highest_volume_put["lastTradeDate_Local"]
                if pd.notnull(last_trade_local):
                    if last_trade_local.date() == today_local_date:
                        total_spent = highest_volume_put['volume'] * highest_volume_put['lastPrice'] * 100
                        formatted_total_spent = format_dollar_amount(total_spent)
                        unusual = highest_volume_put['volume'] > highest_volume_put['openInterest']

                        top_volume_contracts.append({
                            'type': 'PUT',
                            'strike': highest_volume_put['strike'],
                            'volume': highest_volume_put['volume'],
                            'openInterest': highest_volume_put['openInterest'],
                            'date': date,
                            'total_spent': formatted_total_spent,
                            'unusual': unusual
                        })

    # Calculate average strike for visualization
    for date in max_strike_calls.keys():
        if date in max_strike_puts and calls_oi[date] + puts_oi[date] > 0:
            total_oi = calls_oi[date] + puts_oi[date]
            weight_calls = calls_oi[date] / total_oi if total_oi else 0
            weight_puts = puts_oi[date] / total_oi if total_oi else 0

            avg_strike[date] = (
                    (max_strike_calls[date] * weight_calls +
                     second_max_strike_calls[date] * weight_calls +
                     third_max_strike_calls[date] * weight_calls +  #
                     max_strike_puts[date] * weight_puts +
                     second_max_strike_puts[date] * weight_puts +
                     third_max_strike_puts[date] * weight_puts) /  #
                    (3 * (weight_calls + weight_puts))
            )
        else:
            avg_strike[date] = np.nan

    # Sort contracts by volume and configure how many you'd like to display on the plotly graph as annotations
    top_volume_contracts.sort(key=lambda x: x['volume'], reverse=True)
    # Most active options:
    top_volume_contracts = top_volume_contracts[:8]  # <-- toggle this value to change the # of 'Most Active' Options

    # Assuming calls_oi and puts_oi are dictionaries with expiration date keys and OI values.
    expirations = list(calls_oi.keys())

    # Compute absolute differences for each expiration date.
    abs_diffs = np.array([abs(calls_oi.get(exp, 0) - puts_oi.get(exp, 0)) for exp in expirations])
    mean_abs_diff = np.mean(abs_diffs)
    std_abs_diff = np.std(abs_diffs)
    threshold = .95  # Threshold multiplier (can be adjusted)

    # Define minimum and maximum border widths.
    min_width = 1.25
    max_width = 3

    # Pre-calculate the maximum absolute difference (to avoid division by zero).
    max_abs_diff = np.max(abs_diffs) if np.max(abs_diffs) > 0 else 1

    # Prepare arrays for marker line properties.
    calls_line_colors = []
    calls_line_widths = []
    puts_line_colors = []
    puts_line_widths = []

    for exp in expirations:
        call_val = calls_oi.get(exp, 0)
        put_val = puts_oi.get(exp, 0)
        diff = call_val - put_val
        abs_diff = abs(diff)

        # Scale the border width proportionally to how extreme the difference is.
        # The more extreme, the wider the border.
        scaled_width = min_width + (abs_diff / max_abs_diff) * (max_width - min_width)

        # Optionally, only scale for "outlier" differences.
        if abs_diff > (mean_abs_diff + threshold * std_abs_diff):
            width_to_use = scaled_width
        else:
            width_to_use = min_width

        if call_val > put_val:
            # Highlight the call bar with bright green.
            calls_line_colors.append("#29ff30")
            calls_line_widths.append(width_to_use)
            puts_line_colors.append("rgba(0,0,0,0)")
            puts_line_widths.append(0)
        elif put_val > call_val:
            # Highlight the put bar with bright red.
            puts_line_colors.append("#ff2e2e")
            puts_line_widths.append(width_to_use)
            calls_line_colors.append("rgba(0,0,0,0)")
            calls_line_widths.append(0)
        else:
            # If equal, no border on either.
            calls_line_colors.append("rgba(0,0,0,0)")
            calls_line_widths.append(0)
            puts_line_colors.append("rgba(0,0,0,0)")
            puts_line_widths.append(0)

    # Now, add the Plotly bar traces using these computed arrays.

    fig = go.Figure()

    # Bar for Call OI.
    fig.add_trace(go.Bar(
        x=expirations,
        y=[calls_oi.get(exp, 0) for exp in expirations],
        name='Call OI',
        marker=dict(
            color='#708d8b',
            opacity=0.60,
            line=dict(
                color=calls_line_colors,
                width=calls_line_widths
            )
        ),
        showlegend=True,
        hovertemplate='%{y:.3s}<extra></extra>'
    ))

    # Bar for Put OI.
    fig.add_trace(go.Bar(
        x=expirations,
        y=[puts_oi.get(exp, 0) for exp in expirations],
        name='Put OI',
        marker=dict(
            color='#b87d6e',
            opacity=0.60,
            line=dict(
                color=puts_line_colors,
                width=puts_line_widths
            )
        ),
        showlegend=True,
        hovertemplate='%{y:.3s}<extra></extra>'
    ))

    # Add the average strike line
    fig.add_trace(go.Scatter(
        x=list(avg_strike.keys()),  # Sorted expiration dates
        y=list(avg_strike.values()),  # Average strikes per date
        name='Average',
        mode='lines+markers',
        connectgaps=True,
        marker=dict(
            color='#565887',
            size=4,
            symbol='square',  # Change marker shape to square
            line=dict(
                color='black',  # Border color for the markers
                width=1  # Border width for the markers
            )
        ),
        opacity=1,
        yaxis='y2',  # Use secondary y-axis for strike prices
        showlegend=True,
        line=dict(
            color='rgba(40, 40, 43, 1)',
            width=2,
            dash='dashdot'
        ),
        hovertemplate='%{y:.2f}<extra></extra>'
    ))

    # Gather all open interest values for scaling
    all_open_interest = []

    # Collect open interest values from preprocessed calls
    for date, df in calls_data.items():
        if not df.empty:
            all_open_interest.append(df['openInterest'].max())  # Use max OI for scaling

    # Collect open interest values from preprocessed puts
    for date, df in puts_data.items():
        if not df.empty:
            all_open_interest.append(df['openInterest'].max())  # Use max OI for scaling

    # Determine the max open interest for scaling
    max_open_interest = max(all_open_interest) if all_open_interest else 1  # Avoid division by zero

    # Add line plot for max strike calls with scaled markers
    fig.add_trace(go.Scatter(
        x=list(max_strike_calls.keys()),  # Sorted expiration dates
        y=list(max_strike_calls.values()),
        name='Call',
        mode='lines+markers',  # Add markers
        connectgaps=True,
        opacity=0.60,
        yaxis='y2',
        showlegend=True,
        line=dict(color='#75f542', width=2.50),
        marker=dict(
            size=[
                (df['openInterest'].fillna(
                    0).max() / max_open_interest * 20) if not df.empty and max_open_interest > 0 else 5
                for df in calls_data.values()
            ],
            color='#75f542',  # Marker color
            symbol='square',  # Square markers for calls
            line=dict(width=1, color='black')  # Optional: border color for contrast
        ),
        hovertemplate=(
            '<span style="font-family: Arial, sans-serif; font-size:13px;"><b>Strike:</b> $%{y:.2f}<br>'
            '<b>Volume:</b> %{customdata[0]:,}<br>'
            '<b>OI:</b> %{customdata[1]:,}</span><extra></extra>'
        ),
        customdata=[
            (sorted_calls.iloc[0]['volume'], sorted_calls.iloc[0]['openInterest']) if not sorted_calls.empty else (0, 0)
            for sorted_calls in [
                pd.read_csv(os.path.join(calls_dir, filename)).sort_values(by='openInterest', ascending=False).fillna(0)
                for filename in os.listdir(calls_dir)
            ]
        ]
    ))

    fig.add_trace(go.Scatter(
        x=list(second_max_strike_calls.keys()),
        y=list(second_max_strike_calls.values()),
        name='2nd Most-Bought Call',
        mode='lines',
        marker_color='#57f542',
        opacity=.50,
        line=dict(width=2, dash='dot'),
        yaxis='y2',
        showlegend=False,
        hovertemplate='%{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=list(third_max_strike_calls.keys()),
        y=list(third_max_strike_calls.values()),
        name='3rd Most-Bought Call',
        mode='lines',
        marker_color='#25f74f',
        opacity=.25,
        line=dict(width=1.5, dash='dot'),
        yaxis='y2',
        showlegend=False,
        hovertemplate='%{y:.2f}<extra></extra>'
    ))

    # Add line plot for max strike puts with scaled markers
    fig.add_trace(go.Scatter(
        x=list(max_strike_puts.keys()),  # Sorted expiration dates
        y=list(max_strike_puts.values()),
        name='Put',
        mode='lines+markers',  # Add markers
        connectgaps=True,
        opacity=0.60,
        yaxis='y2',
        showlegend=True,
        line=dict(color='#f54242', width=2.50),
        marker=dict(
            size=[
                (df['openInterest'].fillna(
                    0).max() / max_open_interest * 20) if not df.empty and max_open_interest > 0 else 5
                for df in puts_data.values()
            ],
            color='#de3557',  # Marker color
            symbol='square',  # Square markers for puts
            line=dict(width=1, color='black')  # Optional: border color for contrast
        ),
        hovertemplate=(
            '<span style="font-family: Arial, sans-serif; font-size:13px;"><b>Strike:</b> $%{y:.2f}<br>'
            '<b>Volume:</b> %{customdata[0]:,}<br>'
            '<b>OI:</b> %{customdata[1]:,}</span><extra></extra>'
        ),
        customdata=[
            (sorted_puts.iloc[0]['volume'], sorted_puts.iloc[0]['openInterest']) if not sorted_puts.empty else (0, 0)
            for sorted_puts in [
                pd.read_csv(os.path.join(puts_dir, filename)).sort_values(by='openInterest', ascending=False).fillna(0)
                for filename in os.listdir(puts_dir)
            ]
        ]
    ))

    fig.add_trace(go.Scatter(
        x=list(second_max_strike_puts.keys()),
        y=list(second_max_strike_puts.values()),
        name='2nd Most-Bought Put',
        mode='lines',
        marker_color='#d16262',
        opacity=.50,
        line=dict(width=2, dash='dot'),
        yaxis='y2',
        showlegend=False,
        hovertemplate='%{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=list(third_max_strike_puts.keys()),
        y=list(third_max_strike_puts.values()),
        name='3rd Most-Bought Put',
        mode='lines',
        marker_color='#d17b7b',
        opacity=.25,
        line=dict(width=1.5, dash='dot'),
        yaxis='y2',
        showlegend=False,
        hovertemplate='%{y:.2f}<extra></extra>'
    ))

    # Calculate total Volume for calls
    total_call_volume = sum(df['volume'].sum() for df in calls_data.values() if not df.empty)

    # Calculate total Volume for puts
    total_put_volume = sum(df['volume'].sum() for df in puts_data.values() if not df.empty)

    # Format total Volume for display
    formatted_call_volume = f"{int(total_call_volume):,}"
    formatted_put_volume = f"{int(total_put_volume):,}"

    # Determine text color based on which total is higher
    call_color = "#32a852" if total_call_volume > total_put_volume else "#ffffff"  # Green if calls are higher
    put_color = "#de3557" if total_put_volume > total_call_volume else "#ffffff"  # Red if puts are higher

    # Annotation for Net Volume (calls and puts)
    fig.add_annotation(
        text=(
            f"<b>Call Volume: <span style='color:{call_color}'>{formatted_call_volume}</b></span><br>"
            f"<b>Put Volume: <span style='color:{put_color}'>{formatted_put_volume}</b></span>"
        ),
        xref="paper",
        yref="paper",
        x=0.99,  # Keep at far right
        y=1.30,  # Adjust vertical position if needed
        xanchor="right",  # Anchor text on the right
        yanchor="top",
        showarrow=False,
        font=dict(
            family="Arial, sans-serif",
            size=12,  # Increased font size (was 10)
            color="white"
        ),
        align="right",
        bgcolor="#515452",
        bordercolor="#636363",
        borderwidth=10,
        borderpad=8  # Increased padding (was 5)
    )

    # Calculate total premiums for calls
    total_call_premium = sum((df['volume'] * df['lastPrice'] * 100).sum() for df in calls_data.values() if not df.empty)

    # Calculate total premiums for puts
    total_put_premium = sum((df['volume'] * df['lastPrice'] * 100).sum() for df in puts_data.values() if not df.empty)

    # Format total premiums for display
    formatted_call_premium = format_dollar_amount(total_call_premium)
    formatted_put_premium = format_dollar_amount(total_put_premium)

    # Determine text color based on which premium is higher
    call_premium_color = "#32a852" if total_call_premium > total_put_premium else "#ffffff"  # Green if calls are higher
    put_premium_color = "#de3557" if total_put_premium > total_call_premium else "#ffffff"  # Red if puts are higher

    # Annotation for Net Premiums (calls and puts)
    fig.add_annotation(
        text=(
            f"<b>Call Premium: <span style='color:{call_premium_color}'>{formatted_call_premium}</b></span><br>"
            f"<b>Put Premium: <span style='color:{put_premium_color}'>{formatted_put_premium}</b></span>"
        ),
        xref="paper",
        yref="paper",
        x=0.99,
        y=1.15,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        font=dict(
            family="Arial, sans-serif",
            size=12,  # Increased font size (was 10)
            color="white"
        ),
        align="right",
        bgcolor="#515452",
        bordercolor="#636363",
        borderwidth=1,
        borderpad=8  # Increased padding (was 5)
    )

    # Add annotations for the highest volume contracts
    offset_step = 2  # Incremental offset for each subsequent annotation
    for idx, ann in enumerate(top_volume_contracts):
        # Determine color based on option type
        color = '#ff5e00' if ann['type'] == 'PUT' else '#32a852'

        # Change the marker symbol to a diamond
        symbol = 'diamond'

        # Format volume, strike, openInterest, and total_spent with commas
        formatted_volume = f"{int(ann['volume']):,}"
        formatted_strike = f"{int(ann['strike']):,}"
        formatted_total_spent = ann['total_spent']

        # Decide the background color if 'volume' > 'openInterest'
        # Purple hex color example => #8B00FF or #800080
        if ann['unusual']:
            annotation_bg_color = '#2a1c63'  # Purple
        else:
            annotation_bg_color = '#3b3b3b'  # Default

        # Use HTML to format the annotation text
        annotation_text = (
            f"<b><span style='font-size:10px;'>${formatted_strike} <span style='color:{color}'>{ann['type']}</span></span></b><br>"
            f"<span style='font-size:10px;'><span style='color:#cfcfcf'><b>Qty:</span> {formatted_volume}<br></b></span>"
            f"<span style='font-size:10.5px;'><b>{formatted_total_spent}</b></span>"
        )

        # Calculate offset to prevent overlap
        ax_offset = 35 if ann['type'] == 'PUT' else -35  # Right for PUT, Left for CALL
        ay_offset = -35 - (idx * offset_step)  # Vertical offset based on order

        # Point annotations to the strike price on yaxis2
        fig.add_annotation(
            text=annotation_text,
            x=ann['date'],  # Ensure this matches the formatted_date used in preprocessing
            y=ann['strike'],  # Point to the strike price
            yref='y2',  # Reference the y2 axis
            font=dict(
                family='Arial, sans-serif',
                size=8,
                color='#ffffff'
            ),
            bgcolor=annotation_bg_color,  # <-- Use the variable
            showarrow=True,
            arrowhead=0,  # Set to 0 to remove arrowhead shape
            ax=ax_offset,
            ay=ay_offset,
            arrowwidth=1.5,
            bordercolor='#636363',  # Set border color
            borderwidth=1,  # Set border width for thin outline
            borderpad=4  # Padding between the text and the border
        )

        # Add a triangle marker where the annotation is pointing
        fig.add_trace(go.Scatter(
            x=[ann['date']],  # Matches the x-axis dates in the graph
            y=[ann['strike']],  # Ensure this is set to strike price for correct placement
            mode='markers',
            marker=dict(
                size=8,  # Adjust size of the diamond marker
                color=color,  # Use the same color as the annotation for consistency
                symbol=symbol,  # Diamond shape for the marker
                line=dict(width=1, color='#636363')  # Optional: border color for contrast
            ),
            showlegend=False,  # Hide this trace from the legend
            xaxis='x',  # Ensure correct x-axis reference
            yaxis='y2',  # Ensure correct y-axis reference for positioning
            hoverinfo='skip'  # Disable hover text for this marker
        ))

    # Add a horizontal line for the current price
    fig.add_shape(
        type='line',
        x0=0,
        x1=1,
        xref='paper',
        y0=current_price,
        y1=current_price,
        yref='y2',
        line=dict(
            color='#00dbf4',
            width=1.75,
            dash='solid',
        )
    )

    # Add annotation for the current price
    fig.add_annotation(
        text=f'${current_price:.2f}',
        xref='paper',
        x=-0.01,
        y=current_price,
        yref='y2',
        font=dict(
            family='Arial, sans-serif',
            size=14,
            color='#ffffff'
        ),
        bgcolor='#333333',
        showarrow=False
    )

    # Add a dummy trace for the legend
    fig.add_trace(go.Scatter(
        x=[None],  # Use None to keep the trace from appearing on the plot
        y=[None],  # Use None to keep the trace from appearing on the plot
        mode='markers',
        marker=dict(
            size=25,
            color='#3b3b3b',
            symbol='square',
            line=dict(width=1, color='#636363')  # border color
        ),
        name='<span style="color:#10112e">Most Active Options</span>',  # Custom legend text
        showlegend=True
    ))

    dollar_color = "green" if daily_change_dollar > 0 else "red"
    pct_color = "green" if daily_change_pct > 0 else "red"

    title_text = (
        f"<span style='font-size:37px;'>{ticker}</span> "
        f"<span style='font-size:23px;'>({company_name})</span><br>"
        f"<span style='font-size:33px; color:#e8ebe8;'><i>${current_price:.2f}</i></span>"
        f"<span style='font-size:16px;'>"
        f"<span style='color:{dollar_color}; font-style:italic;'>    {daily_change_dollar:+.2f}</span> "
        f"(<span style='color:{pct_color}; font-style:italic;'>{daily_change_pct:+.2f}%</span>)"
        f"</span>"
    )

    # Update layout with consistent settings
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.1,  # Move all the way to the left edge of the plot
            xanchor='left',  # Anchor it on the left side
            y=0.94,
            yanchor='top',
            font=dict(
                size=30,
                family='Times New Roman, serif',
                color='#e8ebe8', # color represents the ticker/company name and the parenthesis () for each plotly graph
                style='italic'
            )
        ),
        # EXPIRATION DATE SECTION/COLORING (x-axis + y-axis configs)
        xaxis=dict(
            title='',
            title_font=dict(size=20, family='Arial, sans-serif', color='#2c3442', style='italic'),
            showgrid=False,
            showline=True,  # Turn on the axis line
            linecolor='#444444',  # Color of the x-axis line which the expiration dates are underneath
            linewidth=1,  # Set line width
            autorange=True,
            tickangle=38,  # Force the x-axis tick labels to display at a 45-degree angle
            tickfont=dict(
                family="Arial, sans-serif",
                size=16,
                color='#e8ebe8' # Color of the Expiration Dates
            )
        ),
        # below is necessary for accurate plotting of the BAR GRAPHS
        yaxis=dict(
            title='',  # Hide the Open Interest title
            showticklabels=False,  # Hide the tick labels
            showgrid=False,  # Hide grid lines
            side='right',
            autorange=True
        ),
        # STRIKE $ SECTION/COLORING (y-axis focused)
        yaxis2=dict(
            title='Strike',
            title_font=dict(
                size=32,
                family="Arial, sans-serif",
                color='#e8ebe8',  # Strike title color
                style='italic'
            ),
            tickfont=dict(
                family="Arial, sans-serif",
                size=19,
                color='#e8ebe8',  # Strike tick numbers color
                style='italic'
            ),
            side='left',
            overlaying='y',
            range=[0, max(max(max_strike_calls.values(), default=0),
                          max(max_strike_puts.values(), default=0),
                          max(avg_strike.values(), default=0),
                          current_price) + 50],
            autorange=True,
            showline=False,
            linecolor='#444444',
            linewidth=0.5,
            showgrid=True,
            gridcolor='rgba(136,136,136,0.10)',  # Grid lines with 50% opacity
            zeroline=True,
            zerolinecolor='rgba(136,136,136,0.25)',
            zerolinewidth=0.5,
            gridwidth=0.5
        ),
        barmode='group',
        plot_bgcolor='#0d0b0c', # Background of the Plotly Graph itself, behind the bar/line graphs
        paper_bgcolor='#0d0b0c', # Border of the Plotly Graph surrounding the outside of the generated graph data
        showlegend=False, # turns the legend on and off ( shows Call OI / Put OI / Average / Call / Put squares etc )
        width=width,
        height=height
    )

    return fig
    pass