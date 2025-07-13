# File: pages/unusual_report.py
import dash
from dash import html
import pandas as pd
import sqlite3
from dash import dash_table

dash.register_page(__name__, path='/unusual', name='Unusual Volume Report')

DB_PATH = 'options.db'

# Utility to fetch unusual volume report data
def get_unusual_report():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT ticker, expiration_date, side, strike, volume, openInterest, unusualness FROM unusual_volume_report ORDER BY expiration_date", conn)
    conn.close()
    return df

layout = html.Div([
    html.H2("Unusual Volume Report"),
    dash_table.DataTable(
        id='unusual-table',
        columns=[{'name': col, 'id': col} for col in get_unusual_report().columns],
        data=get_unusual_report().to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
    )
], style={'padding': '20px'})
