import numpy as np
import dash
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import yfinance as yf
from dash import html, dcc, Input, Output

# Register this file as a Dash page
# Multipage support in app.py will pick up this module
dash.register_page(__name__, path='/', name='viewer')

SQLITE_DB_PATH = 'options.db'

# 1) Fetch all available tickers from the SQLite DB
def get_tickers():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query(
        "SELECT DISTINCT ticker FROM option_chain ORDER BY ticker", conn
    )
    conn.close()
    return df['ticker'].tolist()

# 2) Fetch option chain data for the selected ticker
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
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[{'label': t, 'value': t} for t in get_tickers()],
            value=get_tickers()[0] if get_tickers() else None,
            clearable=False
        ),
        dcc.Checklist(
            id='show-unusual',
            options=[{'label': 'Show Unusual Volume', 'value': 'yes'}],
            value=[],
            inline=True,
            style={'margin': '10px 0'}
        ),
        dcc.Graph(id='oi-graph')
    ], style={'padding': '20px'})

# Callback: update graph when ticker or checkbox changes
@dash.callback(
    Output('oi-graph', 'figure'),
    Input('ticker-dropdown', 'value'),
    Input('show-unusual', 'value')
)
def update_graph(symbol, show_unusual):
    # Fetch live last price via yfinance
    ticker_obj = yf.Ticker(symbol)
    try:
        last_price = ticker_obj.fast_info['last_price']
    except Exception:
        # fallback to most recent close
        last_price = ticker_obj.history(period='1d', interval='1m')['Close'].iloc[-1]

    # 1) Load main option-chain data
    df = get_option_data(symbol)
    labels = df['expiration_date'].dt.strftime('%Y-%m-%d').tolist()
    idx = np.arange(len(df))

    # 2) Combine and normalize OI_OI for marker sizing
    all_oi = np.concatenate([df['call_OI_OI'], df['put_OI_OI']])
    if all_oi.size == 0:
        call_norm = np.zeros(len(df))
        put_norm  = np.zeros(len(df))
    else:
        min_oi, max_oi = all_oi.min(), all_oi.max()
        norm = np.ones_like(all_oi) if max_oi == min_oi else (all_oi - min_oi) / (max_oi - min_oi)
        mid = len(df)
        call_norm = norm[:mid]
        put_norm  = norm[mid:]

    # 3) Map normalized weights to marker sizes
    MIN_SIZE, MAX_SIZE = 30, 500
    call_sizes = MIN_SIZE + call_norm * (MAX_SIZE - MIN_SIZE)
    put_sizes  = MIN_SIZE + put_norm  * (MAX_SIZE - MIN_SIZE)

    # 4) Build base Plotly figure
    fig = go.Figure([
        go.Scatter(
            x=idx, y=df['call_strike_OI'], mode='markers+lines', name='Call Strike',
            marker=dict(
                size=call_sizes,
                symbol='square', sizemode='area', color='lightgrey',
                line=dict(width=2, color='green')
            ),
            customdata=np.stack([df['call_volume_OI'], df['call_OI_OI']], axis=-1),
            hovertemplate='$%{y:.2f} CALL<br>Vol: %{customdata[0]}<br>OI: %{customdata[1]}<extra></extra>'
        ),
        go.Scatter(
            x=idx, y=df['put_strike_OI'], mode='markers+lines', name='Put Strike',
            marker=dict(
                size=put_sizes,
                symbol='square', sizemode='area', color='lightgrey',
                line=dict(width=2, color='red')
            ),
            customdata=np.stack([df['put_volume_OI'], df['put_OI_OI']], axis=-1),
            hovertemplate='$%{y:.2f} PUT<br>Vol: %{customdata[0]}<br>OI: %{customdata[1]}<extra></extra>'
        ),
        go.Bar(
            x=idx, y=df['call_OI_sum'], name='Call OI Sum', yaxis='y2', opacity=0.6,
            hovertemplate='%{y}<extra></extra>'
        ),
        go.Bar(
            x=idx, y=df['put_OI_sum'],  name='Put OI Sum',  yaxis='y2', opacity=0.6,
            hovertemplate='%{y}<extra></extra>'
        )
    ])

    # Add horizontal line for last price
    fig.add_hline(
        y=last_price,
        line=dict(color='blue', dash='dash'),
        annotation_text=f"Last Price: ${last_price:.2f}",
        annotation_position='top right'
    )

    # 5) Conditionally add up to 5 unusual volume annotations
    if 'yes' in show_unusual:
        uv = get_unusual_volume(symbol)
        if not uv.empty:
            pos_map = {lbl: i for i, lbl in enumerate(labels)}
            count = 0
            for _, row in uv.iterrows():
                if count >= 5:
                    break
                key = row['expiration_date'].strftime('%Y-%m-%d')
                if key in pos_map:
                    if row['side'].lower() == 'call':
                        ax, ay = 0, -30   # arrow up, square below
                        border = 'green'
                    else:
                        ax, ay = 0, 30  # arrow down, square above
                        border = 'red'
                    fig.add_annotation(
                        x=pos_map[key], y=row['strike'],
                        text=f"${row['strike']}{row['side'].capitalize()}\nx{row['volume']}",
                        showarrow=True, arrowhead=2, ax=ax, ay=ay,
                        bgcolor='lightgrey', bordercolor=border,
                        borderwidth=2, font=dict(size=11)
                    )
                    count += 1

    # 6) Finalize layout
    fig.update_layout(
        title=f"{symbol} Option Chain  |  Last Price: ${last_price:.2f}",
        xaxis=dict(title='Expiration Date', tickmode='array', tickvals=idx, ticktext=labels),
        yaxis=dict(title='Strike', showline=False, showgrid=False, zeroline=False),
        yaxis2=dict(title='Total Chain OI', overlaying='y', side='right', showline=False, showgrid=False, zeroline=False),
        legend=dict(orientation='h', y=1.05, x=0.3),
        margin=dict(t=80, b=40)
    )
    return fig
