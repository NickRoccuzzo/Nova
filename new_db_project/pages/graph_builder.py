# pages/graph_builder.py

import os
from urllib.parse import quote_plus as urlquote

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go

from db_config import POSTGRES_DB_URL

# ─── DATABASE CONFIG ─────────────────────────────────────────────────────────
DB_USER = os.environ.get("DB_USER", "option_user")
DB_PASS = os.environ.get("DB_PASS", "option_pass")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "tickers")
DATABASE_URL = (
    f"postgresql://{DB_USER}:{urlquote(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
engine = create_engine(DATABASE_URL, future=True)

# Register page
dash.register_page(__name__, path="/graph-builder", name="Option Chain")

# ─── Helper functions ─────────────────────────────────────────────────────────
def human_format(n) -> str:
    try:
        v = float(n)
    except (TypeError, ValueError):
        return ""
    if np.isnan(v):
        return ""
    for unit in ["", "k", "M", "B", "T"]:
        if abs(v) < 1000:
            return f"${v:,.1f}{unit}"
        v /= 1000
    return f"${v:.1f}P"


def make_annotation(label, y, strike, vol, price, side, y_offset=40):
    price = float(price)
    total_spent = vol * price
    color = "green" if side == "CALL" else "red"
    text = (
        f"${strike:.2f} {side}<br>"
        f"{vol:,} contracts<br>"
        f"{human_format(total_spent)}"
    )
    return dict(
        x=label, y=y, xref="x", yref="y1", text=text,
        showarrow=True, arrowhead=2, ax=0,
        ay=-y_offset if side == "CALL" else y_offset,
        bgcolor="rgba(255,255,255,0.9)", bordercolor=color,
        font=dict(color=color)
    )

# ─── Layout ──────────────────────────────────────────────────────────────────
layout = html.Div(
    style={"padding": "2rem", "maxWidth": "1000px", "margin": "auto"},
    children=[
        html.Div([
            dcc.Link('◀ Back to Overview', href='/', style={'marginBottom':'1rem', 'display':'inline-block'}),
        ]),
        html.H1(id="graph-title"),
        dcc.Location(id='url', refresh=False),
        html.Div(
            style={"display": "flex", "gap": "1rem", "marginBottom": "1rem"},
            children=[
                html.Label("Ticker:"),
                dcc.Input(
                    id="ticker-input", type="text", placeholder="e.g. AAPL",
                    value="", persistence=True, persistence_type="session", style={"width": "120px"}
                ),
                html.Label("# Annotations:"),
                html.Div(
                    dcc.Slider(
                        id="num-annotations", min=0, max=10, step=1, value=3,
                        marks={i: str(i) for i in range(11)},
                        persistence=True, persistence_type="session"
                    ), style={"flex": 1}
                ),
                html.Button('Search', id='search-btn', n_clicks=0)
            ],
        ),
        dcc.Graph(id="chain-graph", style={"height": "70vh"})
    ],
)

# ─── Callbacks ────────────────────────────────────────────────────────────────
@callback(
    Output('ticker-input', 'value'),
    Input('url', 'search')
)
def sync_input(search):
    if not search:
        return dash.no_update
    import urllib.parse as up
    params = up.parse_qs(search.lstrip('?'))
    sym = params.get('symbol', [None])[0]
    return sym.upper() if sym else dash.no_update

@callback(
    Output("chain-graph", "figure"),
    Output("graph-title", "children"),
    Input("ticker-input", "value"),
    Input("num-annotations", "value"),
    Input("search-btn", "n_clicks")
)
def update_graph(symbol: str, num_ann: int, n_clicks) -> tuple[go.Figure, str]:
    # Determine symbol
    sym = (symbol or "").strip().upper()
    title = f"{sym} Option-Chain Strikes & OI" if sym else ""
    if not sym:
        return go.Figure(), title

    # 1) Pull enriched metrics
    df = pd.read_sql(
        text("""
            SELECT
              om.expiration_id, e.expiration_date,
              om.call_oi_sum, om.put_oi_sum,
              om.max_call_strike, om.max_call_last_price, om.max_call_volume, om.max_call_oi,
              om.max_put_strike, om.max_put_last_price, om.max_put_volume, om.max_put_oi
            FROM option_metrics om
            JOIN expirations e USING(expiration_id)
            JOIN tickers t     USING(ticker_id)
            WHERE t.symbol = :sym
            ORDER BY e.expiration_date
        """), engine, params={"sym": sym}, parse_dates=["expiration_date"]
    )
    if df.empty:
        return go.Figure(), title

    # labels & base fig
    df["expiry_label"] = df["expiration_date"].dt.strftime("%m/%d/%y")
    x = df["expiry_label"]
    fig = go.Figure()

    # Bars
    fig.add_trace(go.Bar(x=x, y=df["call_oi_sum"], name="Call OI Sum", marker_color="lightgreen", yaxis="y2", opacity=0.6))
    fig.add_trace(go.Bar(x=x, y=df["put_oi_sum"],  name="Put OI Sum",  marker_color="lightcoral", yaxis="y2", opacity=0.6))

    # Marker sizing
    call_oi = df["max_call_oi"].fillna(0).astype(float)
    put_oi  = df["max_put_oi"].fillna(0).astype(float)
    max_io = max(call_oi.max(), put_oi.max(), 1.0)
    call_sizes = np.clip(call_oi / max_io * 50, 10, None)
    put_sizes  = np.clip(put_oi  / max_io * 50, 10, None)

    call_cd = list(zip(df["max_call_volume"].fillna(0).astype(int), df["max_call_last_price"].fillna(0).astype(float), df["max_call_oi"].fillna(0).astype(int)))
    put_cd  = list(zip(df["max_put_volume"].fillna(0).astype(int),  df["max_put_last_price"].fillna(0).astype(float),  df["max_put_oi"].fillna(0).astype(int)))

    # Scatter max strikes
    fig.add_trace(go.Scatter(x=x, y=df["max_call_strike"], mode="lines+markers", name="Max Call Strike",
                             line=dict(color="green", width=2), marker=dict(size=call_sizes, symbol="square", color="green", line=dict(width=1, color="darkgreen")),
                             customdata=call_cd, hovertemplate="<b>Strike:</b> $%{y:.2f}<br><b>Vol:</b> %{customdata[0]:,}<br><b>Last:</b> $%{customdata[1]:.2f}<br><b>OI:</b> %{customdata[2]:,}<extra></extra>", yaxis="y1"))
    fig.add_trace(go.Scatter(x=x, y=df["max_put_strike"],  mode="lines+markers", name="Max Put Strike",
                             line=dict(color="red", width=2),   marker=dict(size=put_sizes,  symbol="square", color="red",   line=dict(width=1, color="darkred")),
                             customdata=put_cd,  hovertemplate="<b>Strike:</b> $%{y:.2f}<br><b>Vol:</b> %{customdata[0]:,}<br><b>Last:</b> $%{customdata[1]:.2f}<br><b>OI:</b> %{customdata[2]:,}<extra></extra>", yaxis="y1"))

    # Unusual annotations
    ue = pd.read_sql(
        text("""
          SELECT u.*
          FROM unusual_events_ranked u
          JOIN tickers t USING(ticker_id)
          WHERE t.symbol = :sym AND u.score_rank <= :num_ann
          ORDER BY u.score_rank
        """), engine, params={"sym": sym, "num_ann": num_ann}, parse_dates=["expiration_date"]
    )
    last_price = float(engine.connect().execute(text("SELECT close FROM price_history WHERE symbol=:sym ORDER BY date DESC LIMIT 1"), {"sym": sym}).scalar() or 0.0)
    df["max_call_volume"] = df["max_call_volume"].fillna(0).astype(int)
    df["max_put_volume"] = df["max_put_volume"].fillna(0).astype(int)
    for _, r in ue.iterrows():
        row = df[df["expiration_id"] == r["expiration_id"]]
        if row.empty: continue
        row = row.iloc[0]
        if r["unusual_max_vol_call"]:
            vol, strike, side = int(row["max_call_volume"]), row["max_call_strike"], "CALL"
        else:
            vol, strike, side = int(row["max_put_volume"]),  row["max_put_strike"],  "PUT"
        if vol == 0 or pd.isna(strike): continue
        ann = make_annotation(row["expiry_label"], strike, strike, vol, last_price, side)
        fig.add_annotation(**ann)

    fig.update_layout(
        title=title, barmode="overlay",
        xaxis=dict(type="category", categoryarray=x, tickangle=45, title="Expiry"),
        yaxis=dict(title="Strike Price", side="left"),
        yaxis2=dict(overlaying="y", side="right", title="Open Interest (bars)", showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=80, b=100)
    )

    return fig, title
