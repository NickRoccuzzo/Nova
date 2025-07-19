# File: /pages/viewer.py

import numpy as np
import dash
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import yfinance as yf
from dash import html, dcc, Input, Output, State, no_update
from option_project_x.apps.utils import make_option_chain_figure

# Register this file as a Dash page
# Multipage support in app.py will pick up this module
dash.register_page(__name__, path='/', name='viewer')

SQLITE_DB_PATH = 'options.db'

# // -- Fetch all available tickers from SQLite
def get_tickers():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query(
        "SELECT DISTINCT ticker FROM option_chain ORDER BY ticker", conn
    )
    conn.close()
    return df['ticker'].tolist()


# // -- Get option chain data
def get_option_data(symbol):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    query = '''
    SELECT
      expiration_date,
      call_strike_OI,
      call_volume_OI,
      call_OI_OI,
      put_strike_OI,
      put_volume_OI,
      put_OI_OI,
      call_OI_sum,
      put_OI_sum
    FROM option_chain
    WHERE ticker = ?
    ORDER BY expiration_date
    '''
    df = pd.read_sql_query(query, conn, params=[symbol])
    conn.close()
    df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.normalize()
    return df


# 3) Fetch unusual volume entries for annotations
def get_unusual_volume(symbol):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    query = '''
    SELECT expiration_date, side, strike, volume
    FROM unusual_volume_report
    WHERE ticker = ?
    '''
    df = pd.read_sql_query(query, conn, params=[symbol])
    conn.close()
    df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.normalize()
    return df


# Page layout function called by Dash Pages
def layout():
    return html.Div([
        html.H2("Option Chain Viewer"),
        dcc.Input(
            id='ticker-input',
            type='text',
            value='',  # ← ensures it’s controlled from the get‑go
            placeholder="Search your ticker...",
            style={'width': '200px', 'marginBottom': '10px'}
        ),
        dcc.Checklist(
            id='show-unusual',
            options=[{'label': 'Show Option Flow', 'value': 'yes'}],
            value=[],
            inline=True,
            style={'margin': '10px 0'}
        ),
        dcc.Graph(id='oi-graph')
    ], style={'padding': '20px'})


@dash.callback(
    Output('ticker-dropdown', 'value'),
    Input('ticker-dropdown', 'search_value'),
    State('ticker-dropdown', 'options'),
    prevent_initial_call=True
)
def select_exact_ticker(search, options):
    """
    If the user’s search string exactly equals one of the option values,
    immediately set that as the dropdown’s value.
    """
    if not search:
        return no_update

    # normalize case so “m” matches “M”
    target = search.strip().upper()
    # look for an exact match in the options’ values
    for opt in options:
        if opt['value'].upper() == target:
            return opt['value']

    # otherwise, do nothing (leave the dropdown open)
    return no_update

# Callback: update graph when ticker or checkbox changes
@dash.callback(
    Output('oi-graph','figure'),
    Input('ticker-input','n_submit'),
    Input('ticker-input','value'),
    Input('show-unusual','value'),
)
def update_graph(n, val, show):
    if not n or not val:
        return go.Figure()
    return make_option_chain_figure(val, 'yes' in show)
