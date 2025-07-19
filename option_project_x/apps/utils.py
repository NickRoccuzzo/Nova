# File: option_project_x/apps/utils.py

import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# Path to your SQLite DB
SQLITE_DB_PATH = 'options.db'


def get_tickers():
    """
    Return a list of all distinct tickers in the option_chain table.
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query(
        "SELECT DISTINCT ticker FROM option_chain ORDER BY ticker",
        conn
    )
    conn.close()
    return df['ticker'].tolist()


def get_option_data(symbol):
    """
    Load the main option‐chain rows for a given symbol.
    """
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

    # normalize dates to midnight
    df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.normalize()
    return df


def get_unusual_volume(symbol):
    """
    Load up to all the unusual volume annotations for a given ticker.
    """
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


def make_option_chain_figure(symbol: str, show_unusual: bool = False) -> go.Figure:
    """
    Build and return the Plotly Figure for a given ticker,
    optionally overlaying up to 5 unusual‐volume annotations.
    """
    # 1) guard and normalize symbol
    symbol = symbol.strip().upper()
    if not symbol or symbol not in get_tickers():
        return go.Figure()

    # 2) fetch last price via yfinance
    ticker_obj = yf.Ticker(symbol)
    try:
        last_price = ticker_obj.fast_info['last_price']
    except Exception:
        hist = ticker_obj.history(period='1d')
        last_price = hist['Close'].iloc[-1] if not hist.empty else None

    # 3) load option‐chain data
    df = get_option_data(symbol)
    labels = df['expiration_date'].dt.strftime('%Y-%m-%d').tolist()
    idx = np.arange(len(df))

    # 4) normalize call/put OI_OI for marker sizing
    all_oi = np.concatenate([df['call_OI_OI'], df['put_OI_OI']])
    if all_oi.size == 0:
        call_norm = np.zeros(len(df))
        put_norm  = np.zeros(len(df))
    else:
        mi, ma = all_oi.min(), all_oi.max()
        if ma == mi:
            norm = np.ones_like(all_oi)
        else:
            norm = (all_oi - mi) / (ma - mi)
        half = len(df)
        call_norm = norm[:half]
        put_norm  = norm[half:]

    MIN_SIZE, MAX_SIZE = 10, 500
    call_sizes = MIN_SIZE + call_norm * (MAX_SIZE - MIN_SIZE)
    put_sizes  = MIN_SIZE + put_norm  * (MAX_SIZE - MIN_SIZE)

    # 5) build the figure
    fig = go.Figure([
        go.Scatter(
            x=idx, y=df['call_strike_OI'],
            mode='markers+lines', name='Call Strike',
            marker=dict(
                size=call_sizes,
                symbol='square', sizemode='area',
                color='lightgrey',
                line=dict(width=2.5, color='green')
            ),
            customdata=np.stack([df['call_volume_OI'], df['call_OI_OI']], axis=-1),
            hovertemplate='$%{y:.2f} CALL<br>Vol: %{customdata[0]}<br>OI: %{customdata[1]}<extra></extra>'
        ),
        go.Scatter(
            x=idx, y=df['put_strike_OI'],
            mode='markers+lines', name='Put Strike',
            marker=dict(
                size=put_sizes,
                symbol='square', sizemode='area',
                color='lightgrey',
                line=dict(width=2.5, color='red')
            ),
            customdata=np.stack([df['put_volume_OI'], df['put_OI_OI']], axis=-1),
            hovertemplate='$%{y:.2f} PUT<br>Vol: %{customdata[0]}<br>OI: %{customdata[1]}<extra></extra>'
        ),
        go.Bar(
            x=idx, y=df['call_OI_sum'],
            name='Call OI Sum', yaxis='y2', opacity=0.45,
            marker_color='#66ff00', hovertemplate='%{y}<extra></extra>'
        ),
        go.Bar(
            x=idx, y=df['put_OI_sum'],
            name='Put OI Sum', yaxis='y2', opacity=0.45,
            marker_color='#8c554a', hovertemplate='%{y}<extra></extra>'
        )
    ])

    # 6) add last‐price line
    if last_price is not None:
        fig.add_hline(
            y=last_price,
            line=dict(color='blue', dash='dash'),
            annotation_text=f"Last Price: ${last_price:.2f}",
            annotation_position='top right'
        )

    # 7) optionally annotate unusual volume
    if show_unusual:
        uv = get_unusual_volume(symbol)
        pos_map = {lbl: i for i, lbl in enumerate(labels)}
        count = 0
        for _, row in uv.iterrows():
            if count >= 5:
                break
            key = row['expiration_date'].strftime('%Y-%m-%d')
            if key not in pos_map:
                continue

            is_call = row['side'].lower() == 'call'
            ay = -30 if is_call else 30
            border = 'green' if is_call else 'red'

            fig.add_annotation(
                x=pos_map[key],
                y=row['strike'],
                text=f"${row['strike']}{row['side'].capitalize()}\nx{row['volume']}",
                showarrow=True,
                arrowhead=2,
                ax=0, ay=ay,
                bgcolor='lightgrey',
                bordercolor=border,
                borderwidth=2,
                font=dict(size=11)
            )
            count += 1

    # 8) finalize layout
    fig.update_layout(
        title=f"{symbol} Option Chain  |  Last Price: ${last_price:.2f}" if last_price else f"{symbol} Option Chain",
        xaxis=dict(title='Expiration Date', tickmode='array', tickvals=idx, ticktext=labels),
        yaxis=dict(title='Strike',       showline=False, showgrid=False, zeroline=False),
        yaxis2=dict(
            title='Total Chain OI', overlaying='y', side='right',
            showline=False, showgrid=False, zeroline=False
        ),
        legend=dict(orientation='h', y=1.05, x=0.3),
        margin=dict(t=80, b=40)
    )

    return fig
