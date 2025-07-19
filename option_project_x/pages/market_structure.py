# File: pages/market_structure.py

import sqlite3
import pandas as pd
import dash
from dash import html, dcc, dash_table, Input, Output, State
import plotly.graph_objects as go

from apps.utils import make_option_chain_figure

dash.register_page(__name__, path="/market-structure", name="Market Structure")

SQLITE_DB_PATH = "options.db"

def load_market_structure():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
          ticker           AS symbol,
          expiration_date,
          structure,
          pct_diff,
          avg_oi,
          days_to_expiry,
          pairedness,
          final_score
        FROM market_structure_report
        ORDER BY final_score DESC
        """,
        conn,
        parse_dates=["expiration_date"]
    )
    conn.close()
    return df

layout = html.Div([
    html.H2("Market Structure Report"),

    # Flex container: left = table, right = preview
    html.Div([
        # Left column: the table
        html.Div(
            dash_table.DataTable(
                id="ms-table",
                columns=[
                    {"name": "Ticker",      "id": "symbol"},
                    {"name": "Expiry",      "id": "expiration_date"},
                    {"name": "Type",        "id": "structure"},
                    {"name": "% Diff",      "id": "pct_diff"},
                    {"name": "Avg OI",      "id": "avg_oi"},
                    {"name": "DTE",         "id": "days_to_expiry"},
                    {"name": "Pairedness",  "id": "pairedness"},
                    {"name": "Final Score", "id": "final_score"},
                ],
                data=load_market_structure().to_dict("records"),
                sort_action="native",
                filter_action="native",
                hidden_columns=[
                    "expiration_date",
                    "avg_oi",
                    "days_to_expiry"
                ],
                style_table={
                    "width": "100%",
                    "overflowX": "auto"
                },
                style_cell={
                    "textAlign": "center",
                    "fontSize": "16px",
                    "padding": "8px"
                },
                style_header={
                    "fontWeight": "bold",
                    "fontSize": "18px"
                },
                style_data_conditional=[
                    {
                        "if": {
                            "filter_query": '{structure} = "BULLISH"',
                            "column_id": "structure"
                        },
                        "backgroundColor": "green",
                        "color": "white",
                        "fontWeight": "bold"
                    },
                    {
                        "if": {
                            "filter_query": '{structure} = "BEARISH"',
                            "column_id": "structure"
                        },
                        "backgroundColor": "red",
                        "color": "white",
                        "fontWeight": "bold"
                    }
                ]
            ),
            style={"flex": "1", "paddingRight": "20px"}
        ),

        # Right column: the preview graph
        html.Div([
            html.H3("Ticker Preview", style={"textAlign": "center"}),
            dcc.Graph(id="ms-preview-graph", config={"displayModeBar": False})
        ], style={"flex": "1", "paddingLeft": "20px"})
    ],
    style={
        "display": "flex",
        "flexDirection": "row",
        "alignItems": "flex-start",
        "marginTop": "20px"
    })

], style={"padding": "20px"})


@dash.callback(
    Output("ms-preview-graph", "figure"),
    Input("ms-table", "active_cell"),
    State("ms-table", "data")
)
def update_preview(active_cell, rows):
    if not active_cell:
        return go.Figure()
    row_idx = active_cell["row"]
    symbol  = rows[row_idx]["symbol"]
    # show_unusual=False for a clean preview; set True if you want flow too
    return make_option_chain_figure(symbol, show_unusual=False)
