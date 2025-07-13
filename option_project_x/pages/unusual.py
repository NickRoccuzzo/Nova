import dash
from dash import html, dash_table, dcc, Input, Output
import pandas as pd
import sqlite3
import numpy as np

# Registering the page
dash.register_page(__name__, path='/unusual', name='Unusual Volume Report')

# SQLite File
default_db = 'options.db'

# Load and preprocess the unusual volume report
def get_unusual_report(db_path=default_db):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT ticker,
               expiration_date,
               side,
               strike,
               volume,
               openInterest,
               unusualness
        FROM unusual_volume_report
        """,
        conn
    )
    conn.close()

    # Compute DTE (business days)
    df['expiration_date'] = pd.to_datetime(df['expiration_date'])
    today = np.datetime64(pd.Timestamp.today().normalize(), 'D')
    exp = df['expiration_date'].values.astype('datetime64[D]')
    df['DTE'] = np.busday_count(today, exp)

    # Format date for display
    df['expiration_date'] = df['expiration_date'].dt.strftime('%Y-%m-%d')

    return df

# Layout
def layout():
    return html.Div([
        html.H2("Unusual Volume Report"),

        # Ticker search input
        dcc.Input(
            id='ticker-search',
            type='text',
            placeholder='Filter by ticker... e.g. AAPL',
            style={'margin-bottom': '10px', 'width': '250px'}
        ),

        # DataTable
        dash_table.DataTable(
            id='unusual-table',
            columns=[
                {'name': 'Ticker', 'id': 'ticker'},
                {'name': 'Expiry', 'id': 'expiration_date'},
                {'name': 'DTE', 'id': 'DTE'},
                {'name': 'Strike', 'id': 'strike'},
                {'name': 'Volume', 'id': 'volume'},
                {'name': 'Open Interest', 'id': 'openInterest'},
                {'name': 'Unusualness', 'id': 'unusualness'},
            ],
            data=[],  # populated by callback
            page_size=20,
            sort_action='native',      # allow sorting by any column
            sort_mode='multi',
            style_table={'overflowX': 'auto', 'maxHeight': '600px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            style_data_conditional=[
                # Color CALL green, PUT red in Strike column
                {
                    'if': {
                        'column_id': 'strike',
                        'filter_query': '{strike} contains "CALL"'
                    },
                    'color': 'green'
                },
                {
                    'if': {
                        'column_id': 'strike',
                        'filter_query': '{strike} contains "PUT"'
                    },
                    'color': 'red'
                },
                # Retain your DTE-based expiration_date formatting
                {
                    'if': {
                        'filter_query': '{DTE} <= 7',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#5402cf',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{DTE} > 7 && {DTE} <= 21',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#ca8cff',
                    'color': 'black'
                },
                {
                    'if': {
                        'filter_query': '{DTE} > 21 && {DTE} <= 42',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#debbfc',
                    'color': 'black'
                },
                {
                    'if': {
                        'filter_query': '{DTE} > 42 && {DTE} <= 64',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#eedefc',
                    'color': 'black'
                },
            ],
        )
    ], style={'padding': '20px'})

# Callback to populate and filter data
@dash.callback(
    Output('unusual-table', 'data'),
    Input('ticker-search', 'value')
)
def update_unusual_data(ticker_search):
    df = get_unusual_report()
    # Combine strike + side into one column
    df['strike'] = df.apply(
        lambda row: f"${row['strike']} {row['side'].upper()}",
        axis=1
    )
    # Filter by ticker search term
    if ticker_search:
        term = ticker_search.strip().upper()
        df = df[df['ticker'].str.upper().str.contains(term)]
    return df.to_dict('records')
