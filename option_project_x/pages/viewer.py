# File: pages/viewer.py
import dash
from dash import html, dcc, Input, Output
import pandas as pd
import sqlite3
import plotly.graph_objects as go

dash.register_page(__name__, path='/', name='Viewer')

DB_PATH = 'options.db'

# Utility to fetch available tickers from the database
def get_tickers():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT DISTINCT ticker FROM option_chain", conn)
    conn.close()
    return df['ticker'].tolist()

# Utility to fetch option chain data for a given symbol
def get_option_data(symbol):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT expiration_date, call_strike_OI, put_strike_OI, call_OI_sum, put_OI_sum
        FROM option_chain
        WHERE ticker = ?
        ORDER BY expiration_date
        """,
        conn,
        params=(symbol,)
    )
    conn.close()
    df['expiration_date'] = pd.to_datetime(df['expiration_date'])
    return df

layout = html.Div([
    html.H2("Option Chain Viewer"),
    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': t, 'value': t} for t in get_tickers()],
        value=get_tickers()[0] if get_tickers() else None,
        clearable=False
    ),
    dcc.Graph(id='oi-graph')
], style={'padding': '20px'})

@dash.callback(
    Output('oi-graph', 'figure'),
    Input('ticker-dropdown', 'value')
)
def update_graph(symbol):
    df = get_option_data(symbol)
    fig = go.Figure()
    # Line: strike price of highest-OI contract
    fig.add_trace(go.Scatter(
        x=df['expiration_date'],
        y=df['call_strike_OI'],
        mode='lines+markers', marker_symbol='square', name='Call Strike'
    ))
    fig.add_trace(go.Scatter(
        x=df['expiration_date'],
        y=df['put_strike_OI'],
        mode='lines+markers', marker_symbol='square', name='Put Strike'
    ))
    # Bar: total OI
    fig.add_trace(go.Bar(
        x=df['expiration_date'], y=df['call_OI_sum'], name='Total Call OI', marker_color='green', yaxis='y2'
    ))
    fig.add_trace(go.Bar(
        x=df['expiration_date'], y=df['put_OI_sum'], name='Total Put OI', marker_color='red', yaxis='y2'
    ))
    fig.update_layout(
        title=f"{symbol} Strikes & Open Interest by Expiration",
        xaxis=dict(title='Expiration Date'),
        yaxis=dict(title='Strike Price'),
        yaxis2=dict(title='Total Chain OI', overlaying='y', side='right'),
        legend=dict(orientation='h', y=1.05, x=0.3)
    )
    return fig
