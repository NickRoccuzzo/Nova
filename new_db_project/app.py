import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server

app.layout = html.Div([
    dbc.Nav(
        [
            dbc.NavLink(page["name"], href=page["path"], active="exact")
            for page in dash.page_registry.values()
            if page["module"].startswith("pages.")
        ],
        pills=True,
        className="mb-4"
    ),
    dash.page_container
])

if __name__ == "__main__":
    app.run(debug=True)