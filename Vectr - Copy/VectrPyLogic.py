# ─── MODULES ──────────────────────────────────────────────────────────────────
import os
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
import yfinance as yf


# ─── Helpers ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import zoneinfo


# ────────────────────────────────────────────────────────────────
# 🎨 PERSONAL DESIGN CONTROL PANEL
#    Change a hex once → every figure updates automatically
# ────────────────────────────────────────────────────────────────
BAR_CALL_COLOR    = "#5a916d"     # forest‑green
BAR_PUT_COLOR     = "#875656"     # wine‑red
LINE_CALL_COLOR   = "#75f542"
LINE_PUT_COLOR    = "#f54242"
AVG_STRIKE_COLOR  = "#565887"
BACKGROUND_COLOR  = "#0d0b0c"     # swap to "#a8a8a8" for light theme
TEXT_PRIMARY      = "#e8ebe8"     # tick‑labels / titles on dark bg
TEXT_SECONDARY    = "#01234a"     # tick‑labels / titles on light bg

# ——————————————————————————————————————————————
# Re‑usable Plotly layout template
# ——————————————————————————————————————————————
BASE_LAYOUT = {
    "plot_bgcolor":  BACKGROUND_COLOR,
    "paper_bgcolor": BACKGROUND_COLOR,
    "showlegend":    True,                    # turn legend back on if you want
    "legend": {                               # optional – keep or remove
        "x":0.5, "y":1.10, "xanchor":"center", "yanchor":"top",
        "orientation":"h",
        "font":{"family":"Arial, sans-serif","size":10,"color":TEXT_PRIMARY},
    },

    # x‑axis
    "xaxis": {
        "title": "",
        "showgrid": False,
        "showline": True,
        "linecolor": "#444444",
        "linewidth": 1,
        "tickangle": 38,
        "tickfont": {"family":"Arial, sans-serif","size":16,"color":TEXT_PRIMARY},
    },

    # y‑axis (bars)
    "yaxis": {
        "title": "", "showticklabels": False, "showgrid": False,
        "side":"right", "autorange":True,
    },

    # secondary y‑axis (strike prices)
    "yaxis2": {
        "title": "Strike",
        "title_font": {"family":"Arial, sans-serif","size":32,"color":TEXT_PRIMARY},
        "tickfont":  {"family":"Arial, sans-serif","size":19,"color":TEXT_PRIMARY},
        "side":         "left",
        "overlaying":   "y",
        "showline":     False,
        "linecolor":    "#444444",
        "linewidth":    0.5,
        "showgrid":     True,
        "gridcolor":    "rgba(136,136,136,0.10)",
        "zeroline":     True,
        "zerolinecolor":"rgba(136,136,136,0.25)",
        "zerolinewidth":0.5,
        "gridwidth":    0.5,
    },

    "barmode": "group",
}


local_tz = zoneinfo.ZoneInfo("America/New_York")

@dataclass
class SideSummary:
    oi: int
    strikes: Tuple[float, float, float]      # top‑3 by OI, padded with 0s
    vol_row: Optional[pd.Series]             # None if no volume data

@dataclass
class MostActiveAnn:
    kind: str
    strike: float
    volume: int
    oi: int
    exp: str
    total: str
    unusual: bool

    def annotation(self, idx: int) -> dict:
        color = "#ff5e00" if self.kind == "PUT" else "#32a852"
        bg    = "#2a1c63" if self.unusual else "#3b3b3b"
        ax    = 35 if self.kind == "PUT" else -35
        ay    = -35 - idx*2
        return {
            "text": (
                f"<b><span style='font-size:10px;'>${int(self.strike):,} "
                f"<span style='color:{color}'>{self.kind}</span></span></b><br>"
                f"<span style='font-size:10px;'><span style='color:#cfcfcf'><b>Qty:</span> "
                f"{int(self.volume):,}<br></b></span>"
                f"<span style='font-size:10.5px;'><b>{self.total}</b></span>"
            ),
            "x": self.exp,
            "y": self.strike,
            "yref": "y2",
            "bgcolor": bg,
            "showarrow": True,
            "arrowhead": 0,
            "ax": ax,
            "ay": ay,
            "arrowwidth": 1.5,
            "bordercolor": "#636363",
            "borderwidth": 1,
            "borderpad": 4,
            "font": {"family":"Arial, sans-serif","size":8,"color":"#ffffff"},
        }

    def marker(self) -> dict:
        return {
            "x": [self.exp],
            "y": [self.strike],
            "mode":"markers",
            "marker":{
                "size":8,
                "color":"#ff5e00" if self.kind=="PUT" else "#32a852",
                "symbol":"diamond",
                "line":{"width":1,"color":"#636363"}
            },
            "yaxis":"y2",
            "hoverinfo":"skip",
            "showlegend":False,
        }


def summarise_side(df: pd.DataFrame) -> SideSummary:
    """
    Given a CALL or PUT dataframe for *one* expiry, return:
      • total open‑interest
      • top‑3 strikes by open‑interest
      • the row with the highest volume (or None)
    """
    df = df.copy()

    # Sum OI
    oi_sum = int(df['openInterest'].sum())

    # Top‑3 strikes by OI (pad with zeros so length is always 3)
    top3 = (
        df.sort_values('openInterest', ascending=False)
          .get('strike')
          .head(3)
          .tolist() + [0, 0, 0]
    )[:3]

    # Row with the highest volume (ignore NaNs)
    vol_row = (
        df.loc[df['volume'].idxmax()]          # type: ignore[arg-type]
        if df['volume'].notna().any()
        else None
    )

    return SideSummary(oi=oi_sum, strikes=tuple(top3), vol_row=vol_row)

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


def find_top_volume_contracts(
    data_dicts: dict[str, pd.DataFrame],
    today: date | None = None,
) -> list[dict]:
    """
    Inspect every expiry dataframe (CALLS or PUTS) in `data_dicts`
    and return a list with at most one dict per side, containing the
    highest‑volume row *for today only*.

    Each dict has keys:
        type, strike, volume, openInterest, date, total_spent, unusual
    """
    if today is None:
        today = datetime.now(local_tz).date()

    results: list[dict] = []

    for side, dfs in data_dicts.items():           # e.g. {"CALL": calls_data, ...}
        # Flatten all rows into one big DF so we can idxmax once
        big_df = (
            pd.concat([df.assign(_exp=exp) for exp, df in dfs.items()], ignore_index=True)
            if dfs else pd.DataFrame()
        )
        if big_df.empty or big_df["volume"].isna().all():
            continue

        row = big_df.loc[big_df["volume"].idxmax()]

        last_trade = row.get("lastTradeDate_Local")
        if pd.isna(last_trade) or last_trade.date() != today:
            continue

        spent = row["volume"] * row["lastPrice"] * 100

        results.append({
            "type":         side,
            "strike":       row["strike"],
            "volume":       int(row["volume"]),
            "openInterest": int(row["openInterest"]),
            "date":         row["_exp"],
            "total_spent":  format_dollar_amount(spent),
            "unusual":      row["volume"] > row["openInterest"],
        })

    return results


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


def gather_options_data(ticker: str) -> dict:
    """
    Return the raw aggregates used by the Plotly logic—no figure creation.
    Keys returned:
      calls_oi, puts_oi,
      max_strike_calls / second_max_strike_calls / third_max_strike_calls,
      max_strike_puts  / second_max_strike_puts  / third_max_strike_puts,
      avg_strike,
      current_price, company_name
    """

    # ── Locate CALLS / PUTS directories ───────────────────────────────────────
    base_dir  = os.getcwd()
    calls_dir = os.path.join(base_dir, ticker, "CALLS")
    puts_dir  = os.path.join(base_dir, ticker, "PUTS")

    calls_data = preprocess_dates(calls_dir, "CALLS")
    puts_data  = preprocess_dates(puts_dir,  "PUTS")

    # ── Dicts for OI & strikes ───────────────────────────────────────────────
    calls_oi, puts_oi = {}, {}

    max_strike_calls, second_max_strike_calls, third_max_strike_calls = {}, {}, {}
    max_strike_puts,  second_max_strike_puts,  third_max_strike_puts  = {}, {}, {}

    # ── Iterate once over CALLS and PUTS ─────────────────────────────────────
    for side, data_dict in (("CALL", calls_data), ("PUT", puts_data)):
        for date, df in data_dict.items():
            if df.empty:
                continue

            summary = summarise_side(df)  # ← new helper

            if side == "CALL":
                calls_oi[date] = summary.oi
            else:
                puts_oi[date]  = summary.oi

            d_max, d_second, d_third = summary.strikes
            if side == "CALL":
                max_strike_calls[date]        = d_max
                second_max_strike_calls[date] = d_second
                third_max_strike_calls[date]  = d_third
            else:
                max_strike_puts[date]         = d_max
                second_max_strike_puts[date]  = d_second
                third_max_strike_puts[date]   = d_third

    # ── Vectorised average‑strike calculation (runs **once**) ────────────────
    df_avg = (
        pd.DataFrame({
            "calls_oi": pd.Series(calls_oi),
            "puts_oi":  pd.Series(puts_oi),
            "c1": pd.Series(max_strike_calls),
            "c2": pd.Series(second_max_strike_calls),
            "c3": pd.Series(third_max_strike_calls),
            "p1": pd.Series(max_strike_puts),
            "p2": pd.Series(second_max_strike_puts),
            "p3": pd.Series(third_max_strike_puts),
        })
        .fillna(0)
    )

    tot_oi   = df_avg["calls_oi"] + df_avg["puts_oi"]
    w_calls  = np.where(tot_oi > 0, df_avg["calls_oi"] / tot_oi, 0)
    w_puts   = np.where(tot_oi > 0, df_avg["puts_oi"]  / tot_oi, 0)

    sum_calls = df_avg[["c1", "c2", "c3"]].sum(axis=1)
    sum_puts  = df_avg[["p1", "p2", "p3"]].sum(axis=1)

    avg_strike = dict(zip(
        df_avg.index,
        np.where(tot_oi > 0, (w_calls * sum_calls + w_puts * sum_puts) / 3, np.nan)
    ))

    # ── Current price & company name ─────────────────────────────────────────
    stock          = yf.Ticker(ticker)
    hist           = stock.history(period="1d")
    current_price  = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
    company_name   = stock.info.get("longName", "N/A")

    # ── Return all aggregates ────────────────────────────────────────────────
    return {
        "calls_oi": calls_oi,
        "puts_oi": puts_oi,
        "max_strike_calls":   max_strike_calls,
        "second_max_strike_calls": second_max_strike_calls,
        "third_max_strike_calls":  third_max_strike_calls,
        "max_strike_puts":    max_strike_puts,
        "second_max_strike_puts":  second_max_strike_puts,
        "third_max_strike_puts":   third_max_strike_puts,
        "avg_strike": avg_strike,
        "current_price": current_price,
        "company_name":  company_name,
        "calls_data": calls_data,
        "puts_data": puts_data,
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

    data = gather_options_data(ticker)
    calls_oi = data["calls_oi"]
    puts_oi = data["puts_oi"]
    max_strike_calls = data["max_strike_calls"]
    second_max_strike_calls = data["second_max_strike_calls"]
    third_max_strike_calls = data["third_max_strike_calls"]
    max_strike_puts = data["max_strike_puts"]
    second_max_strike_puts = data["second_max_strike_puts"]
    third_max_strike_puts = data["third_max_strike_puts"]
    avg_strike = data["avg_strike"]

    # Safely compute the daily change percentage if there's at least 2 rows
    if len(price_data) > 1:
        prev_close = price_data['Close'].iloc[-2]
        daily_change_dollar = current_price - prev_close
        daily_change_pct = (daily_change_dollar / prev_close) * 100
    else:
        daily_change_dollar = 0
        daily_change_pct = 0

    # Preprocess and sort calls and puts data
    calls_data = data["calls_data"]  # new keys you add
    puts_data = data["puts_data"]

    top_volume_contracts = find_top_volume_contracts(
        {"CALL": calls_data, "PUT": puts_data}
    )
    top_volume_contracts.sort(key=lambda x: x["volume"], reverse=True)
    top_volume_contracts = top_volume_contracts[:8]

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
            color=BAR_CALL_COLOR,
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
            color=BAR_PUT_COLOR,
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
            color=AVG_STRIKE_COLOR,
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
        line=dict(color=LINE_CALL_COLOR, width=2.50),
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
            (
                df.sort_values("openInterest", ascending=False).iloc[0]["volume"],
                df.sort_values("openInterest", ascending=False).iloc[0]["openInterest"]
            ) if not df.empty else (0, 0)
            for df in calls_data.values()
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
        line=dict(color=LINE_PUT_COLOR, width=2.50),
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
            (
                df.sort_values("openInterest", ascending=False).iloc[0]["volume"],
                df.sort_values("openInterest", ascending=False).iloc[0]["openInterest"]
            ) if not df.empty else (0, 0)
            for df in puts_data.values()
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

    # ------------------------------------------------------------------
    # “Most active” contract annotations + diamond markers
    # ------------------------------------------------------------------
    for i, raw in enumerate(top_volume_contracts):
        ma = MostActiveAnn(
            kind=raw["type"],
            strike=raw["strike"],
            volume=raw["volume"],
            oi=raw["openInterest"],
            exp=raw["date"],
            total=raw["total_spent"],
            unusual=raw["unusual"],
        )
        fig.add_annotation(**ma.annotation(i))
        fig.add_trace(go.Scatter(**ma.marker()))

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

    # ------------------------------------------------------------------
    # Apply shared layout + per‑figure overrides
    # ------------------------------------------------------------------
    fig.update_layout(
        BASE_LAYOUT | {
            "width": width,
            "height": height,

            # dynamic title
            "title": {
                "text": title_text,
                "x": 0.10,
                "xanchor": "left",
                "y": 0.94,
                "yanchor": "top",
                "font": {"family": "Times New Roman, serif", "size": 30, "color": "#e8ebe8"},
            },
        }
    )

    return fig