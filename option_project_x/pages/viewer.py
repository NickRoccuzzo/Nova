# File: pages/viewer.py
import numpy as np
import dash
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
dash.register_page(__name__, path='/', name='viewer')


SQLITE_DB_PATH = 'options.db'


# Fetches all available tickers from the SQLite DB
def get_tickers():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT DISTINCT ticker FROM option_chain", conn)
    conn.close()
    return df['ticker'].tolist()


# Fetches the option chains associated with those tickers
def get_option_data(symbol):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    query = """
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
    """
    df = pd.read_sql_query(
        query,
        conn,
        params=[symbol]
    )
    conn.close()

    # normalize the date if it’s still a string
    df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.normalize()
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
    # 1) load & normalize date as before
    df = get_option_data(symbol)
    df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.normalize()
    labels = df['expiration_date'].dt.strftime('%Y-%m-%d')
    idx = np.arange(len(df))

    # 2) build a single array of all OI_OI values
    all_oi = np.concatenate([
        df['call_OI_OI'].to_numpy(),
        df['put_OI_OI'].to_numpy()
    ])

    # 3) normalize to [0,1]
    #    if you have outliers, you can use log‐scaling too:
    # log_oi = np.log1p(all_oi)
    # norm = (log_oi - log_oi.min()) / (log_oi.max() - log_oi.min())
    norm_oi = (all_oi - all_oi.min()) / (all_oi.max() - all_oi.min())

    # split back into call vs put normalized
    n = len(df)
    call_norm = norm_oi[:n]
    put_norm = norm_oi[n:]

    # 4) map [0,1] → marker sizes [min_size, max_size]
    MIN, MAX = 10, 500
    call_sizes = MIN + call_norm * (MAX - MIN)
    put_sizes = MIN + put_norm * (MAX - MIN)

    # — build the figure —
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=idx, y=df['call_strike_OI'],
        mode='markers+lines', name='Call Strike',
        marker=dict(
            size=call_sizes,
            symbol='square',
            sizemode='area',
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        customdata=np.stack([df['call_volume_OI'], df['call_OI_OI']], axis=-1),
        hovertemplate=(
            '$%{y:.2f} CALL<br>'
            'Vol: %{customdata[0]}<br>'
            'OI: %{customdata[1]}<extra></extra>'
        )
    ))

    fig.add_trace(go.Scatter(
        x=idx, y=df['put_strike_OI'],
        mode='markers+lines', name='Put Strike',
        marker=dict(
            size=put_sizes,
            symbol='square',
            sizemode='area',
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        customdata=np.stack([df['put_volume_OI'], df['put_OI_OI']], axis=-1),
        hovertemplate=(
            '$%{y:.2f} PUT<br>'
            'Vol: %{customdata[0]}<br>'
            'OI: %{customdata[1]}<extra></extra>'
        )
    ))

    # bars unchanged
    fig.add_trace(go.Bar(
        x=idx, y=df['call_OI_sum'], name='Call OI Sum',
        yaxis='y2', hovertemplate='%{y}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        x=idx, y=df['put_OI_sum'],  name='Put OI Sum',
        yaxis='y2', hovertemplate='%{y}<extra></extra>'
    ))

    # layout as before
    fig.update_layout(
        title=f"{symbol} Option Chain",
        xaxis=dict(tickmode='array', tickvals=idx, ticktext=labels, title='Expiration Date'),
        yaxis=dict(title='Strike'),
        yaxis2=dict(title='Total Chain OI', overlaying='y', side='right'),
        legend=dict(orientation='h', y=1.05, x=0.3)
    )

    return fig