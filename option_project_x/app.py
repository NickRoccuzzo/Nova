# File: app.py

import dash
from dash import Dash, html

app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    suppress_callback_exceptions=True,
)
server = app.server

nav = html.Nav([
    html.A("Viewer",         href="/",                className="nav-link"),
    html.A("Unusual Vol.",   href="/unusual",         className="nav-link"),
    html.A("Market Struct.", href="/market-structure", className="nav-link"),
], style={"display": "flex", "gap": "1rem", "margin": "20px"})

app.layout = html.Div([
    html.H1("Options Dashboard", style={"textAlign": "center"}),

    # Render your nav bar
    nav,

    # Then the page container
    dash.page_container
])

if __name__ == "__main__":
    app.run(debug=True)

