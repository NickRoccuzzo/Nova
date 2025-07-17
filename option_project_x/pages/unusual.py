import dash
from dash import html, dash_table, dcc, Input, Output
import pandas as pd
import sqlite3
import numpy as np
from dash_table.Format import Format, Group, Scheme, Symbol

# Registering the page
dash.register_page(__name__, path='/unusual', name='Unusual Volume Report')

# SQLite File
default_db = 'options.db'

# Load and preprocess the unusual volume report
def get_unusual_report(db_path=default_db):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            ticker,
            expiration_date,
            side,
            strike,
            volume,
            openInterest,
            unusualness,
            total_spent
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
            value='',  # initialize as controlled input
            placeholder='Filter by ticker... e.g. AAPL',
            style={'marginBottom': '10px', 'width': '250px'}
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
                {'name': 'OI', 'id': 'openInterest'},
                {'name': 'Vol:OI Ratio', 'id': 'unusualness'},
                {
                    'name': 'Total Spent',
                    'id': 'total_spent',
                    'type': 'numeric',
                    'format': Format(
                        symbol=Symbol.yes,
                        symbol_prefix='$',
                        group=Group.yes,
                        group_delimiter=',',
                        scheme=Scheme.fixed,
                        precision=2
                    )
                },
            ],
            data=[],  # populated by callback
            page_size=20,
            sort_action='native',      # allow sorting by any column
            sort_mode='multi',
            sort_by=[
                {'column_id': 'DTE', 'direction': 'asc'}
            ],
            # Table dimensions and scrolling
            style_table={
                'width': '100%',
                'height': '700px',
                'overflowX': 'auto',
                'overflowY': 'auto'
            },
            # Cell styling and font size
            style_cell={
                'textAlign': 'left',
                'padding': '8px',
                'fontSize': '14px',
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_header={
                'backgroundColor': 'lightgrey',
                'fontWeight': 'bold',
                'fontSize': '16px'
            },
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
                # DTE-based expiration_date formatting
                {
                    'if': {
                        'filter_query': '{DTE} <= 3',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#2f2ab8',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{DTE} > 3 && {DTE} <= 5',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#423dba',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{DTE} > 5 && {DTE} <= 10',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#5754ba',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{DTE} > 10 && {DTE} <= 20',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#6765b8',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{DTE} > 20 && {DTE} <= 30',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#7674b5',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{DTE} > 30 && {DTE} <= 40',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#8b8ab5',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{DTE} > 40 && {DTE} <= 50',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#9f9eb8',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{DTE} > 50 && {DTE} <= 60',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#adadb8',
                    'color': 'black'
                },
                {
                    'if': {
                        'filter_query': '{DTE} > 60 && {DTE} <= 90',
                        'column_id': 'expiration_date'
                    },
                    'backgroundColor': '#dedee0',
                    'color': 'black'
                },
                {
                    'if': {'column_id': 'total_spent', 'filter_query': '{total_spent} > 500000'},
                    'backgroundColor': 'gold', 'color': 'black', 'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'total_spent', 'filter_query': '{total_spent} > 1000000'},
                    'backgroundColor': '#e1ff00', 'color': 'black', 'fontWeight': 'bold'
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
