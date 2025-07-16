# File: pages/market_structure.py
import sqlite3
import pandas as pd
import dash
from dash import html, dash_table

dash.register_page(__name__, path="/market-structure", name="Market Structure")

SQLITE_DB_PATH = "options.db"

def load_market_structure():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
          ticker   AS symbol,
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
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center"},
        style_header={"fontWeight": "bold"},
    )
], style={"padding": "20px"})