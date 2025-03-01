

import json
import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd

from NovaSync import (
    build_play_dictionaries,
    build_oi_volume_dictionaries
)

app = dash.Dash(__name__)
server = app.server

# --------------------------------------------------------------------
# 1) LOAD DATA
# --------------------------------------------------------------------
summary_file = r"C:\Users\DEFAULT.DESKTOP-30IV20T\PycharmProjects\pythonProject\NOVASYNC\summary_results2.json"

# A) From build_play_dictionaries
(
    bullish_plays_dict,
    bearish_plays_dict,
    unusual_activity_dict,
    money_flow_dict,
    call_flow_dict,
    put_flow_dict
) = build_play_dictionaries(summary_file)

# B) From build_oi_volume_dictionaries
(
    most_volume_puts_dict,
    most_volume_calls_dict,
    highest_ratio_calls_oi_dict,
    highest_ratio_puts_oi_dict,
    whale_call_dict,
    whale_put_dict,
    whale_call_oi_dict,
    whale_put_oi_dict
) = build_oi_volume_dictionaries(summary_file)

# Load the raw JSON to reference date-level data
with open(summary_file, 'r') as f:
    raw_data = json.load(f)
all_tickers_data = raw_data.get("all_tickers", {})


# --------------------------------------------------------------------
# 2) HELPER FUNCTIONS
# --------------------------------------------------------------------

def flatten_bullish_or_bearish(plays_dict, y_col="score"):
    """
    For bullish_plays_dict or bearish_plays_dict (and similarly unusual_activity_dict, etc.)
    we produce a DataFrame with Ticker on x-axis and the chosen measure on y-axis (like 'score').

    We also include company_name, current_price, etc. in the DF columns for reference.
    The 'score' might be replaced by 'unusual_contracts_count' or 'total_unusual_spent'
    depending on the dictionary at hand.
    """
    rows = []
    for ticker, info in plays_dict.items():
        rows.append({
            "Ticker": ticker,
            "CompanyName": info.get("company_name", "Unknown"),
            "CurrentPrice": info.get("current_price", 0),
            "Value": info.get(y_col, 0)  # e.g. 'score' or 'unusual_contracts_count' etc.
        })
    return pd.DataFrame(rows)


def make_simple_bar(df, title):
    """
    For the 'plays' dictionaries (bullish, bearish, etc.), we only have Ticker-level data
    (no date-level breakdown). We'll do Ticker on X and 'Value' on Y.

    color='Ticker' just to show variety, or you can omit color if you prefer 1 color.

    This is where you'd add color_discrete_sequence if you have custom hex codes.
    e.g. color_discrete_sequence=["#123456", "#abcdef", ...]
    """
    if df.empty:
        return px.bar(title=f"{title} (No data)")
    fig = px.bar(
        df,
        x="Ticker",
        y="Value",
        color="Ticker",
        title=title
        # color_discrete_sequence=[ ... ] if you want custom hex codes
    )
    return fig


# ------ VOLUME flattening (already shown in earlier examples) ------
def flatten_volume_dict(ticker_dict, all_data, volume_key="calls_volume"):
    """
    Flatten date-level volume (calls_volume or puts_volume) for each Ticker.
    ticker_dict is e.g. most_volume_calls_dict, whale_call_dict, etc.
    """
    rows = []
    for ticker, info in ticker_dict.items():
        # For date-level data we go back to all_data[ticker][volume_key]
        raw_ticker_info = all_data.get(ticker, {})
        vol_dict = raw_ticker_info.get(volume_key, {})  # e.g. {"02/28/25": 82, ...}

        company_name = info.get("company_name", "Unknown")
        current_price = info.get("current_price", 0)

        for expiration_date, val in vol_dict.items():
            rows.append({
                "Ticker": ticker,
                "Expiration": expiration_date,
                "Value": val,
                "CompanyName": company_name,
                "CurrentPrice": current_price
            })
    return pd.DataFrame(rows)


def flatten_oi_dict(ticker_dict, all_data, oi_key="calls_oi"):
    """
    Flatten date-level OI (calls_oi or puts_oi) for each Ticker.
    """
    rows = []
    for ticker, info in ticker_dict.items():
        raw_ticker_info = all_data.get(ticker, {})
        oi_dict = raw_ticker_info.get(oi_key, {})

        company_name = info.get("company_name", "Unknown")
        current_price = info.get("current_price", 0)

        for expiration_date, val in oi_dict.items():
            rows.append({
                "Ticker": ticker,
                "Expiration": expiration_date,
                "Value": val,
                "CompanyName": company_name,
                "CurrentPrice": current_price
            })
    return pd.DataFrame(rows)


def make_grouped_bar(df, title):
    """
    For the 'OI/Volume' dictionaries, we have date-level data.
    We'll do Ticker on X, Value on Y, color by Expiration.
    """
    if df.empty:
        return px.bar(title=f"{title} (No data)")
    fig = px.bar(
        df,
        x="Ticker",
        y="Value",
        color="Expiration",
        barmode="group",
        title=title
        # color_discrete_sequence=[ ... ] if custom colors
    )
    return fig


# --------------------------------------------------------------------
# 3) BUILD FIGURES FOR "build_play_dictionaries" (6 dicts)
# --------------------------------------------------------------------

# a) Bullish plays -> flatten with y_col="score"
df_bullish = flatten_bullish_or_bearish(bullish_plays_dict, y_col="score")
fig_bullish = make_simple_bar(df_bullish, "Bullish Plays (Score)")

# b) Bearish plays -> flatten with y_col="score"
df_bearish = flatten_bullish_or_bearish(bearish_plays_dict, y_col="score")
fig_bearish = make_simple_bar(df_bearish, "Bearish Plays (Score)")

# c) Unusual activity -> flatten with y_col="unusual_contracts_count"
df_unusual = flatten_bullish_or_bearish(unusual_activity_dict, y_col="unusual_contracts_count")
fig_unusual = make_simple_bar(df_unusual, "Most Unusual Activity (Contracts)")

# d) Money flow -> flatten with y_col="total_unusual_spent"
df_moneyflow = flatten_bullish_or_bearish(money_flow_dict, y_col="total_unusual_spent")
fig_moneyflow = make_simple_bar(df_moneyflow, "Money Flow (Total Unusual Spent)")

# e) Call flow -> flatten with y_col="calls_to_puts_ratio"
df_callflow = flatten_bullish_or_bearish(call_flow_dict, y_col="calls_to_puts_ratio")
fig_callflow = make_simple_bar(df_callflow, "Call Flow Ratio (Calls/Puts)")

# f) Put flow -> flatten with y_col="puts_to_calls_ratio"
df_putflow = flatten_bullish_or_bearish(put_flow_dict, y_col="puts_to_calls_ratio")
fig_putflow = make_simple_bar(df_putflow, "Put Flow Ratio (Puts/Calls)")

# --------------------------------------------------------------------
# 4) BUILD FIGURES FOR "build_oi_volume_dictionaries" (8 dicts)
# --------------------------------------------------------------------

# 4.1) most_volume_calls_dict -> flatten with calls_volume
df_mvcalls = flatten_volume_dict(most_volume_calls_dict, all_tickers_data, volume_key="calls_volume")
fig_mvcalls = make_grouped_bar(df_mvcalls, "Most Volume Calls")

# 4.2) most_volume_puts_dict -> flatten with puts_volume
df_mvputs = flatten_volume_dict(most_volume_puts_dict, all_tickers_data, volume_key="puts_volume")
fig_mvputs = make_grouped_bar(df_mvputs, "Most Volume Puts")

# 4.3) highest_ratio_calls_oi_dict -> flatten with calls_oi
df_hr_calls_oi = flatten_oi_dict(highest_ratio_calls_oi_dict, all_tickers_data, oi_key="calls_oi")
fig_hr_calls_oi = make_grouped_bar(df_hr_calls_oi, "Highest Ratio Calls OI")

# 4.4) highest_ratio_puts_oi_dict -> flatten with puts_oi
df_hr_puts_oi = flatten_oi_dict(highest_ratio_puts_oi_dict, all_tickers_data, oi_key="puts_oi")
fig_hr_puts_oi = make_grouped_bar(df_hr_puts_oi, "Highest Ratio Puts OI")

# 4.5) whale_call_dict -> flatten calls_volume
df_whale_calls = flatten_volume_dict(whale_call_dict, all_tickers_data, volume_key="calls_volume")
fig_whale_calls = make_grouped_bar(df_whale_calls, "Whale Call Volume")

# 4.6) whale_put_dict -> flatten puts_volume
df_whale_puts = flatten_volume_dict(whale_put_dict, all_tickers_data, volume_key="puts_volume")
fig_whale_puts = make_grouped_bar(df_whale_puts, "Whale Put Volume")

# 4.7) whale_call_oi_dict -> flatten calls_oi
df_whale_call_oi = flatten_oi_dict(whale_call_oi_dict, all_tickers_data, oi_key="calls_oi")
fig_whale_call_oi = make_grouped_bar(df_whale_call_oi, "Whale Call OI")

# 4.8) whale_put_oi_dict -> flatten puts_oi
df_whale_put_oi = flatten_oi_dict(whale_put_oi_dict, all_tickers_data, oi_key="puts_oi")
fig_whale_put_oi = make_grouped_bar(df_whale_put_oi, "Whale Put OI")

# --------------------------------------------------------------------
# 5) MASTER TABLE
# --------------------------------------------------------------------
# Flatten some top-level fields into a single DataFrame so we can display
# them in a search/sort table.
master_rows = []
for ticker, info in all_tickers_data.items():
    master_rows.append({
        "Ticker": ticker,
        "Company": info.get("company_name", ""),
        "CurrentPrice": info.get("current_price", ""),
        "Score": info.get("score", ""),
        "UnusualCount": info.get("unusual_contracts_count", ""),
        "TotalUnusualSpent": info.get("total_unusual_spent", ""),
        "calls_volume": json.dumps(info.get("calls_volume", {})),
        "puts_volume": json.dumps(info.get("puts_volume", {})),
        "calls_oi": json.dumps(info.get("calls_oi", {})),
        "puts_oi": json.dumps(info.get("puts_oi", {})),
    })
master_df = pd.DataFrame(master_rows)

master_table = dash_table.DataTable(
    columns=[{"name": c, "id": c} for c in master_df.columns],
    data=master_df.to_dict('records'),
    filter_action="native",
    sort_action="native",
    page_size=10,
    style_table={'overflowX': 'auto'}
)

# --------------------------------------------------------------------
# 6) APP LAYOUT
# --------------------------------------------------------------------
# We'll do two major sections:
# A) A 3×2 grid for the 6 'play' dictionaries
# B) A 4×2 grid for the 8 'OI/Volume' dictionaries
# C) Master table
app.layout = html.Div([
    html.H1("Options Analysis Dashboard", style={'textAlign': 'center'}),

    html.H2("A) Score- & Flow-Based Dicts (from build_play_dictionaries)"),
    # 3 rows x 2 columns
    html.Div([
        # Row 1
        html.Div([
            html.Div([dcc.Graph(figure=fig_bullish)], style={'width': '50%'}),
            html.Div([dcc.Graph(figure=fig_bearish)], style={'width': '50%'}),
        ], style={'display': 'flex'}),

        # Row 2
        html.Div([
            html.Div([dcc.Graph(figure=fig_unusual)], style={'width': '50%'}),
            html.Div([dcc.Graph(figure=fig_moneyflow)], style={'width': '50%'}),
        ], style={'display': 'flex'}),

        # Row 3
        html.Div([
            html.Div([dcc.Graph(figure=fig_callflow)], style={'width': '50%'}),
            html.Div([dcc.Graph(figure=fig_putflow)], style={'width': '50%'}),
        ], style={'display': 'flex'}),
    ]),

    html.H2("B) OI & Volume Dicts (from build_oi_volume_dictionaries)"),
    # 4 rows x 2 columns = 8 charts
    html.Div([
        # Row 1
        html.Div([
            html.Div([dcc.Graph(figure=fig_mvcalls)], style={'width': '50%'}),
            html.Div([dcc.Graph(figure=fig_mvputs)], style={'width': '50%'}),
        ], style={'display': 'flex'}),

        # Row 2
        html.Div([
            html.Div([dcc.Graph(figure=fig_hr_calls_oi)], style={'width': '50%'}),
            html.Div([dcc.Graph(figure=fig_hr_puts_oi)], style={'width': '50%'}),
        ], style={'display': 'flex'}),

        # Row 3
        html.Div([
            html.Div([dcc.Graph(figure=fig_whale_calls)], style={'width': '50%'}),
            html.Div([dcc.Graph(figure=fig_whale_puts)], style={'width': '50%'}),
        ], style={'display': 'flex'}),

        # Row 4
        html.Div([
            html.Div([dcc.Graph(figure=fig_whale_call_oi)], style={'width': '50%'}),
            html.Div([dcc.Graph(figure=fig_whale_put_oi)], style={'width': '50%'}),
        ], style={'display': 'flex'}),
    ]),

    html.Hr(),
    html.H2("C) Master Searchable Table of All Tickers"),
    master_table
])

if __name__ == "__main__":
    app.run_server(debug=True)
