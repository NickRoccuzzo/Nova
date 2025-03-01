

import json
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

from dash.dependencies import Input, Output, State

from NovaSync import (
    build_play_dictionaries,
    build_oi_volume_dictionaries
)

app = dash.Dash(__name__)
server = app.server

summary_file = r"C:\Users\DEFAULT.DESKTOP-30IV20T\PycharmProjects\pythonProject\NOVASYNC\summary_results2.json"

# ----------------------------------------------------
# 1) LOAD ALL DICTIONARIES
# ----------------------------------------------------
(
    bullish_plays_dict,
    bearish_plays_dict,
    unusual_activity_dict,
    money_flow_dict,
    call_flow_dict,
    put_flow_dict
) = build_play_dictionaries(summary_file)

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

# Also load the full raw JSON
with open(summary_file, "r") as f:
    raw_data = json.load(f)
all_tickers_data = raw_data.get("all_tickers", {})


# ----------------------------------------------------
# 2) FLATTEN + CHART HELPERS
# ----------------------------------------------------
def flatten_play_dict(dct, value_key="score"):
    """Flatten single-bar dictionaries from build_play_dictionaries (like bullish_plays_dict)."""
    rows = []
    for ticker, info in dct.items():
        rows.append({
            "Ticker": ticker,
            "Value": info.get(value_key, 0),
            "CompanyName": info.get("company_name", "Unknown"),
            "CurrentPrice": info.get("current_price", 0)
        })
    return pd.DataFrame(rows)


def make_single_bar(df, title, color_hex):
    """
    Make a single-bar chart, using the user-provided color scheme.
    (One bar per Ticker)
    """
    if df.empty:
        return px.bar(title=f"{title} (No data)")

    fig = px.bar(
        df,
        x="Ticker",
        y="Value",
        title=title,
        color_discrete_sequence=[color_hex],
        height=600  # Increase height a bit
    )
    return fig


def flatten_volume_dict(dct, all_data, volume_key="calls_volume"):
    """Flatten date-level calls_volume or puts_volume."""
    rows = []
    for ticker, info in dct.items():
        vol_map = all_data.get(ticker, {}).get(volume_key, {})
        for expiration_date, val in vol_map.items():
            rows.append({
                "Ticker": ticker,
                "Expiration": expiration_date,
                "Value": val,
                "CompanyName": info.get("company_name", "Unknown"),
                "CurrentPrice": info.get("current_price", 0)
            })
    return pd.DataFrame(rows)


def flatten_oi_dict(dct, all_data, oi_key="calls_oi"):
    """Flatten date-level calls_oi or puts_oi."""
    rows = []
    for ticker, info in dct.items():
        oi_map = all_data.get(ticker, {}).get(oi_key, {})
        for expiration_date, val in oi_map.items():
            rows.append({
                "Ticker": ticker,
                "Expiration": expiration_date,
                "Value": val,
                "CompanyName": info.get("company_name", "Unknown"),
                "CurrentPrice": info.get("current_price", 0)
            })
    return pd.DataFrame(rows)


def make_grouped_bar(df, title):
    """
    Grouped bar chart for multi-exp data: x=Ticker, color=Expiration, default colors.
    """
    if df.empty:
        return px.bar(title=f"{title} (No data)", height=600)
    fig = px.bar(
        df,
        x="Ticker",
        y="Value",
        color="Expiration",
        barmode="group",
        title=title,
        height=600
    )
    return fig


# ----------------------------------------------------
# 3) COLOR SCHEME FOR "build_play_dictionaries"
# ----------------------------------------------------
COLOR_BULLISH = "#4bf046"  # green
COLOR_BEARISH = "#eb3847"  # red
COLOR_UNUSUAL = "#a88fe3"  # purple
COLOR_MONEYFLOW = "#9ebcf7"  # blue
COLOR_CALLFLOW = "#aae38f"  # green (lighter)
COLOR_PUTFLOW = "#d18c8c"  # red (lighter)

# We'll keep default colors for the OI & Volume charts.

# ----------------------------------------------------
# 4) DICTIONARY CONFIG (Dropdown Options)
# ----------------------------------------------------
dict_options = {
    # build_play_dictionaries
    "Bullish Plays": {
        "dictionary": bullish_plays_dict,
        "flatten_func": flatten_play_dict,
        "flatten_kwargs": {"value_key": "score"},
        "chart_func": make_single_bar,
        "chart_kwargs": {"color_hex": COLOR_BULLISH},
        "title": "Bullish Plays (Score)"
    },
    "Bearish Plays": {
        "dictionary": bearish_plays_dict,
        "flatten_func": flatten_play_dict,
        "flatten_kwargs": {"value_key": "score"},
        "chart_func": make_single_bar,
        "chart_kwargs": {"color_hex": COLOR_BEARISH},
        "title": "Bearish Plays (Score)"
    },
    "Most Unusual Activity": {
        "dictionary": unusual_activity_dict,
        "flatten_func": flatten_play_dict,
        "flatten_kwargs": {"value_key": "unusual_contracts_count"},
        "chart_func": make_single_bar,
        "chart_kwargs": {"color_hex": COLOR_UNUSUAL},
        "title": "Most Unusual Activity"
    },
    "Money Flow": {
        "dictionary": money_flow_dict,
        "flatten_func": flatten_play_dict,
        "flatten_kwargs": {"value_key": "total_unusual_spent"},
        "chart_func": make_single_bar,
        "chart_kwargs": {"color_hex": COLOR_MONEYFLOW},
        "title": "Money Flow (Total Unusual Spent)"
    },
    "Call Flow Ratio": {
        "dictionary": call_flow_dict,
        "flatten_func": flatten_play_dict,
        "flatten_kwargs": {"value_key": "calls_to_puts_ratio"},
        "chart_func": make_single_bar,
        "chart_kwargs": {"color_hex": COLOR_CALLFLOW},
        "title": "Call Flow Ratio (Calls/Puts)"
    },
    "Put Flow Ratio": {
        "dictionary": put_flow_dict,
        "flatten_func": flatten_play_dict,
        "flatten_kwargs": {"value_key": "puts_to_calls_ratio"},
        "chart_func": make_single_bar,
        "chart_kwargs": {"color_hex": COLOR_PUTFLOW},
        "title": "Put Flow Ratio (Puts/Calls)"
    },

    # build_oi_volume_dictionaries
    "Most Volume Calls": {
        "dictionary": most_volume_calls_dict,
        "flatten_func": flatten_volume_dict,
        "flatten_kwargs": {"all_data": all_tickers_data, "volume_key": "calls_volume"},
        "chart_func": make_grouped_bar,
        "chart_kwargs": {},
        "title": "Most Volume Calls"
    },
    "Most Volume Puts": {
        "dictionary": most_volume_puts_dict,
        "flatten_func": flatten_volume_dict,
        "flatten_kwargs": {"all_data": all_tickers_data, "volume_key": "puts_volume"},
        "chart_func": make_grouped_bar,
        "chart_kwargs": {},
        "title": "Most Volume Puts"
    },
    "Highest Ratio Calls OI": {
        "dictionary": highest_ratio_calls_oi_dict,
        "flatten_func": flatten_oi_dict,
        "flatten_kwargs": {"all_data": all_tickers_data, "oi_key": "calls_oi"},
        "chart_func": make_grouped_bar,
        "chart_kwargs": {},
        "title": "Highest Ratio Calls OI"
    },
    "Highest Ratio Puts OI": {
        "dictionary": highest_ratio_puts_oi_dict,
        "flatten_func": flatten_oi_dict,
        "flatten_kwargs": {"all_data": all_tickers_data, "oi_key": "puts_oi"},
        "chart_func": make_grouped_bar,
        "chart_kwargs": {},
        "title": "Highest Ratio Puts OI"
    },
    "Whale Call Volume": {
        "dictionary": whale_call_dict,
        "flatten_func": flatten_volume_dict,
        "flatten_kwargs": {"all_data": all_tickers_data, "volume_key": "calls_volume"},
        "chart_func": make_grouped_bar,
        "chart_kwargs": {},
        "title": "Whale Call Volume"
    },
    "Whale Put Volume": {
        "dictionary": whale_put_dict,
        "flatten_func": flatten_volume_dict,
        "flatten_kwargs": {"all_data": all_tickers_data, "volume_key": "puts_volume"},
        "chart_func": make_grouped_bar,
        "chart_kwargs": {},
        "title": "Whale Put Volume"
    },
    "Whale Call OI": {
        "dictionary": whale_call_oi_dict,
        "flatten_func": flatten_oi_dict,
        "flatten_kwargs": {"all_data": all_tickers_data, "oi_key": "calls_oi"},
        "chart_func": make_grouped_bar,
        "chart_kwargs": {},
        "title": "Whale Call OI"
    },
    "Whale Put OI": {
        "dictionary": whale_put_oi_dict,
        "flatten_func": flatten_oi_dict,
        "flatten_kwargs": {"all_data": all_tickers_data, "oi_key": "puts_oi"},
        "chart_func": make_grouped_bar,
        "chart_kwargs": {},
        "title": "Whale Put OI"
    }
}

dropdown_opts = [{"label": k, "value": k} for k in dict_options.keys()]

# ----------------------------------------------------
# 5) LAYOUT
# ----------------------------------------------------
app.layout = html.Div([
    html.H1("Options Analysis Dashboard", style={"textAlign": "center"}),

    html.Div([
        # Center the dropdown
        html.Div(
            dcc.Dropdown(
                id="dict-dropdown",
                options=dropdown_opts,
                value="Bullish Plays",
                clearable=False,
                style={"width": "400px", "margin": "0 auto"}  # center + set width
            ),
            style={"textAlign": "center"}
        )
    ]),

    # The main chart
    dcc.Graph(id="dict-graph", style={"height": "700px"}),  # or "600px", up to you

    html.Hr(),

    # Ticker search bar
    html.H2("Ticker Search"),
    html.Div([
        dcc.Input(
            id="ticker-input",
            type="text",
            placeholder="Enter Ticker (e.g. EDR)",
            style={"marginRight": "8px"}
        ),
        html.Button("Search", id="search-button", n_clicks=0),
    ], style={"marginBottom": "12px"}),

    html.Div(id="search-result")
])


# ----------------------------------------------------
# 6) CALLBACKS
# ----------------------------------------------------

# A) Dictionary selection callback
@app.callback(
    Output("dict-graph", "figure"),
    Input("dict-dropdown", "value")
)
def update_graph(selected_key):
    cfg = dict_options[selected_key]
    # Flatten
    df = cfg["flatten_func"](cfg["dictionary"], **cfg["flatten_kwargs"])
    # Make figure
    fig = cfg["chart_func"](df, cfg["title"], **cfg["chart_kwargs"])
    return fig


# B) Ticker search callback
@app.callback(
    Output("search-result", "children"),
    Input("search-button", "n_clicks"),
    State("ticker-input", "value")
)
def search_ticker(n_clicks, ticker_value):
    if n_clicks < 1 or not ticker_value:
        return ""

    # 1) Convert ticker_value to uppercase (assuming your JSON keys are uppercase)
    ticker_value = ticker_value.strip().upper()
    # 2) Check if it is in all_tickers_data
    if ticker_value in all_tickers_data:
        # 3) Retrieve the raw JSON for that ticker
        data_obj = all_tickers_data[ticker_value]
        # 4) Return it as a string
        pretty_str = json.dumps(data_obj, indent=4)
        return html.Pre(pretty_str)
    else:
        return html.Div(f"No data found for ticker '{ticker_value}'", style={"color": "red"})


if __name__ == "__main__":
    app.run_server(debug=True)
