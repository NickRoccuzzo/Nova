# graph_builder.py
import os

import pandas as pd
from sqlalchemy import create_engine, text

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# ─── DATABASE SETUP ─────────────────────────────────────────────────────────────
DB_URI = os.getenv("GRAPH_DB_URI",
                   "postgresql://option_user:option_pass@localhost:5432/tickers")
engine = create_engine(DB_URI, future=True)

# ─── METRIC OPTIONS ────────────────────────────────────────────────────────────
X_OPTIONS = [
    {"label": "Call OI Total",     "value": "call_oi_sum"},
    {"label": "Put OI Total",      "value": "put_oi_sum"},
    {"label": "Call Volume Total", "value": "call_vol_sum"},
    {"label": "Put Volume Total",  "value": "put_vol_sum"},
    {"label": "Call IV Total",     "value": "call_iv_sum"},
    {"label": "Put IV Total",      "value": "put_iv_sum"},
]

Y_OPTIONS = [
    {"label": "Max OI Call",     "value": "max_oi_call"},
    {"label": "Max OI Put",      "value": "max_oi_put"},
    {"label": "2nd Max OI Call", "value": "second_oi_call"},
    {"label": "2nd Max OI Put",  "value": "second_oi_put"},
    {"label": "3rd Max OI Call", "value": "third_oi_call"},
    {"label": "3rd Max OI Put",  "value": "third_oi_put"},
    {"label": "Max Vol Call",     "value": "max_vol_call"},
    {"label": "Max Vol Put",      "value": "max_vol_put"},
    {"label": "2nd Max Vol Call", "value": "second_vol_call"},
    {"label": "2nd Max Vol Put",  "value": "second_vol_put"},
    {"label": "3rd Max Vol Call", "value": "third_vol_call"},
    {"label": "3rd Max Vol Put",  "value": "third_vol_put"},
]

# ─── DASH SETUP ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Option‐Chain Metrics Explorer"

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def load_tickers():
    """Fetch all symbols for dropdown."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT symbol FROM tickers ORDER BY symbol")).all()
    return [r[0] for r in rows]

# New helper: fetch scenario stats

def get_scenario_stats(symbol: str) -> pd.DataFrame:
    """Grab the latest EMA scenario and return metrics for a given symbol."""
    sql = """
    SELECT *
      FROM latest_ticker_stats
     WHERE symbol = :symbol
    """
    df = pd.read_sql(text(sql), engine, params={"symbol": symbol})
    return df

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
app.layout = html.Div([
    html.H1("Option‐Chain Metrics Explorer"),

    html.Div([
        html.Label("Ticker"),
        dcc.Dropdown(
            id="ticker-dropdown",
            options=[{"label": sym, "value": sym}
                     for sym in load_tickers()],
            placeholder="Select a ticker…",
            clearable=False
        ),
    ], style={"width": "20%", "display": "inline-block", "verticalAlign": "top"}),

    html.Div([
        html.Label("X-axis (Totals: bars)"),
        dcc.Checklist(
            id="x-metrics",
            options=X_OPTIONS,
            value=["call_oi_sum", "put_oi_sum"],  # defaults
            inline=True
        ),
        html.Br(),
        html.Label("Y-axis (Strikes/Vol: lines)"),
        dcc.Checklist(
            id="y-metrics",
            options=Y_OPTIONS,
            value=["max_oi_call", "max_oi_put"],  # defaults
            inline=True
        ),
    ], style={"width": "75%", "display": "inline-block", "paddingLeft": "2%"}),

    dcc.Graph(id="metrics-graph", style={"height": "70vh"}),
    html.Div(id="scenario-info", style={"padding": "1em", "fontFamily": "monospace"}),
])


def query_option_metrics(symbol: str) -> pd.DataFrame:
    """Grab option_metrics for all expirations of a given symbol."""
    sql = """
    SELECT
      e.expiration_date::date,
      om.*
    FROM option_metrics om
    JOIN tickers t      ON om.ticker_id = t.ticker_id
    JOIN expirations e  ON om.expiration_id = e.expiration_id
    WHERE t.symbol = :symbol
    ORDER BY e.expiration_date
    """
    df = pd.read_sql(text(sql), engine, params={"symbol": symbol})
    return df


# ─── CALLBACK ────────────────────────────────────────────────────────────────
@app.callback(
    Output("metrics-graph", "figure"),
    Input("ticker-dropdown", "value"),
    Input("x-metrics", "value"),
    Input("y-metrics", "value"),
)
def update_graph(symbol, x_metrics, y_metrics):
    if not symbol:
        return go.Figure()

    df = query_option_metrics(symbol)
    if df.empty:
        return go.Figure().add_annotation(text="No data for that ticker")

    df["exp_str"] = pd.to_datetime(df["expiration_date"]).dt.strftime("%Y-%m-%d")

    fig = go.Figure()
    # bars on primary y-axis
    for col in x_metrics:
        fig.add_trace(go.Bar(
            x=df["exp_str"],
            y=df[col],
            name=col.replace("_", " ").title(),
            opacity=0.6,
        ))

    # lines on secondary y-axis
    for col in y_metrics:
        fig.add_trace(go.Scatter(
            x=df["exp_str"],
            y=df[col],
            name=col.replace("_", " ").title(),
            mode="lines+markers",
            yaxis="y2",
        ))

    # layout with two y-axes
    fig.update_layout(
        title=f"{symbol} Option Metrics by Expiration",
        xaxis_title="Expiration Date",
        yaxis=dict(title="Totals (bars)"),
        yaxis2=dict(
            title="Strikes/Vol (lines)",
            overlaying="y",
            side="right"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # force the axis to be categorical
    fig.update_layout(
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=list(df["exp_str"]),
            tickangle=45,
            tickfont=dict(size=10),
            tickmode="array",
            tickvals=list(df["exp_str"]),
            ticktext=list(df["exp_str"])
        ),
        margin=dict(t=40, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


@app.callback(
    Output("scenario-info", "children"),
    Input("ticker-dropdown", "value")
)
def update_scenario_info(symbol):
    if not symbol:
        return ""

    df = get_scenario_stats(symbol)
    if df.empty:
        return html.Div("No scenario data available", style={"color": "gray"})

    row = df.iloc[0]
    return html.Div([
        html.P(f"Current scenario: {row['current_scenario']}{''}"),
        html.P(f"Avg return: {row['avg_return_for_scenario']:.2f}%"),
        html.P(f"Bull%: {row['bull_percent_for_scenario']:.1f}%  Bear%: {row['bear_percent_for_scenario']:.1f}%"),
        html.P(f"Occurrences: {row['num_occurrences_for_scenario']}{''}")
    ])

# ─── BOOTSTRAP ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
