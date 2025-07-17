# File: app.py

import dash
from dash import Dash, html

# Initialize the Dash app with multipage support
app = Dash(__name__, use_pages=True, pages_folder="pages", suppress_callback_exceptions=True)
server = app.server

# App layout renders whichever page is active
app.layout = html.Div([
    html.H1("Options Dashboard", style={"textAlign": "center"}),
    dash.page_container
])

if __name__ == "__main__":
    app.run(debug=True)

