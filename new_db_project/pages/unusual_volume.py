# pages/unusual_volume.py

import dash
import pandas as pd
from sqlalchemy import create_engine, text
from dash import html, dcc, dash_table, Input, Output, State
import plotly.express as px

from db_config import POSTGRES_DB_URL

# ─── Register the page ─────────────────────────────────────────────────────────
dash.register_page(__name__, path="/unusual-volume", name="Unusual Volume")

# ─── Load data ─────────────────────────────────────────────────────────────────
engine = create_engine(POSTGRES_DB_URL)

# 1) Full ticker-level feed
df = pd.read_sql("SELECT * FROM unusual_volume_report", engine)

# 2) Top-5 industries by max_adj_score
df_ind = pd.read_sql(
    "SELECT industry_name, max_adj_score "
    "FROM industry_unusual_report "
    "ORDER BY max_adj_score DESC "
    "LIMIT 5",
    engine,
)

# 3) For those top-5 industries, grab their top 5 tickers & scores
inds = df_ind["industry_name"].tolist()
df_top_ticks = pd.read_sql(
    text("""
      SELECT i.industry_name, u.symbol, u.adj_score
      FROM unusual_volume_report u
      JOIN tickers    t ON t.symbol      = u.symbol
      JOIN industries i ON i.industry_id = t.industry_id
      WHERE i.industry_name = ANY(:inds)
      ORDER BY i.industry_name, u.adj_score DESC
    """),
    engine,
    params={"inds": inds},
)

# 4) Build a hover‐string for each industry
hover_map = {}
for name, grp in df_top_ticks.groupby("industry_name"):
    top5 = grp.head(5)
    hover_map[name] = "<br>".join(f"{sym}: {score:.2f}"
                                  for sym, score in zip(top5["symbol"], top5["adj_score"]))

df_ind["ticker_list"] = df_ind["industry_name"].map(hover_map)

# 5) Build the horizontal bar chart with custom hovertemplate
fig_ind = px.bar(
    df_ind,
    x="max_adj_score",
    y="industry_name",
    orientation="h",
    custom_data=["ticker_list"],
    labels={"max_adj_score": "Max Adj Score", "industry_name": "Industry"},
    title="Top 5 Industries by Unusual-Volume Adjusted Score",
)
fig_ind.update_layout(margin=dict(l=40, r=20, t=50, b=20))
fig_ind.update_traces(
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Max Score: %{x:.2f}<br><br>"
        "Top Tickers:<br>%{customdata[0]}<extra></extra>"
    )
)

# ─── Page layout ───────────────────────────────────────────────────────────────
layout = html.Div(
    style={"padding": "2rem", "maxWidth": "1200px", "margin": "auto"},
    children=[
        html.H2("Unusual Volume Report"),
        html.Div(
            style={"display": "flex", "gap": "2rem", "alignItems": "flex-start"},
            children=[
                # ─ Left: DataTable ────────────────────────────
                html.Div(
                    dash_table.DataTable(
                        id="unv-table",
                        columns=[
                            {"name": "Ticker",         "id": "symbol"},
                            {"name": "Expiry",         "id": "expiration_date", "type": "datetime"},
                            {"name": "Metric",         "id": "metric_col"},
                            {"name": "Raw Score",      "id": "raw_score",       "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "Strike",         "id": "strike",          "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "Side",           "id": "side"},
                            {"name": "Last Price",     "id": "last_price",      "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "Days to Expiry", "id": "days_to_expiry",  "type": "numeric"},
                            {"name": "Moneyness",      "id": "moneyness",       "type": "numeric", "format": {"specifier": ".4f"}},
                            {"name": "Adj. Score",     "id": "adj_score",       "type": "numeric", "format": {"specifier": ".2f"}},
                        ],
                        data=df.to_dict("records"),
                        sort_action="native",
                        filter_action="native",
                        page_size=20,
                        row_selectable="single",
                        style_data_conditional=[
                            {"if": {"filter_query": "{adj_score} > 0"},   "backgroundColor": "lightgreen"},
                            {"if": {"filter_query": "{adj_score} < 0"},   "backgroundColor": "lightcoral"},
                            {"if": {"filter_query": "-0.0001 < {adj_score} < 0.0001"}, "backgroundColor": "lightgray"},
                        ],
                        style_cell={"textAlign": "center", "padding": "0.5rem"},
                        style_header={"fontWeight": "bold"},
                    ),
                    style={
                        "flex": "0 1 65%",      # ~65% width
                        "maxHeight": "75vh",    # cap height
                        "overflowY": "auto",
                        "overflowX": "auto",
                    },
                ),
                # ─ Right: Bar Chart ───────────────────────────
                html.Div(
                    dcc.Graph(
                        id="unv-ind-bar",
                        figure=fig_ind,
                        style={"height": "75vh"}   # match table height
                    ),
                    style={
                        "flex": "0 1 35%",      # ~35% width
                    },
                ),
            ],
        ),
        # Hidden for redirects
        dcc.Location(id="unv-loc"),
    ],
)

# ─── Callback: drill into graph_builder ──────────────────────────────────────
@dash.callback(
    Output("unv-loc", "href"),
    Input("unv-table", "selected_rows"),
    State("unv-table", "data"),
)
def redirect_to_graph(selected_rows, table_data):
    if not selected_rows:
        return dash.no_update
    idx    = selected_rows[0]
    symbol = table_data[idx]["symbol"]
    return f"/graph-builder?symbol={symbol}"
