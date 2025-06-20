import json
import threading
from pathlib import Path

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import update_option_chains_in_db

# ── Paths & Globals ─────────────────────────────────────────────────────────────
HERE          = Path(__file__).parent
TICKERS_FILE  = HERE / "tickers.json"
PROGRESS_FILE = HERE / "progress.json"
LOG_LINES     = 100

ingest_thread = None

# ── Load tickers structure ─────────────────────────────────────────────────────
with open(TICKERS_FILE) as f:
    tickers_tree = json.load(f)

# ── Build the expandable tree with optional search expansion ───────────────────
def build_tree(search_symbol=None):
    try:
        nested = json.loads(TICKERS_FILE.read_text())
        prog   = json.loads(PROGRESS_FILE.read_text())
    except Exception:
        nested, prog = {}, {}

    search_symbol = search_symbol.strip().upper() if search_symbol else None
    nodes = []

    for sector, industries in nested.items():
        # Determine if this sector should be expanded
        sector_open = False
        if search_symbol:
            for entries in industries.values():
                for ent in entries:
                    sym = ent["symbol"] if isinstance(ent, dict) else ent
                    if sym.upper() == search_symbol:
                        sector_open = True
                        break
                if sector_open:
                    break

        sector_ts = prog.get("sectors", {}).get(sector, {}).get("last_timestamp", "—")
        ind_nodes = []

        for ind, entries in industries.items():
            # Determine if this industry should be expanded
            ind_open = False
            if search_symbol:
                for ent in entries:
                    sym = ent["symbol"] if isinstance(ent, dict) else ent
                    if sym.upper() == search_symbol:
                        ind_open = True
                        break

            ind_ts = prog.get("industries", {}).get(ind, {}).get("last_timestamp", "—")
            ticker_items = []

            for ent in entries:
                sym = ent["symbol"] if isinstance(ent, dict) else ent
                tick_ts = prog.get("tickers", {}).get(sym, {}).get("last_timestamp", "—")
                if search_symbol and sym.upper() == search_symbol:
                    # Highlight the searched ticker
                    ticker_items.append(html.Li(f"{sym} (last: {tick_ts})",
                                                style={"fontWeight": "bold", "color": "red"}))
                else:
                    ticker_items.append(html.Li(f"{sym} (last: {tick_ts})"))

            ind_nodes.append(
                html.Details(
                    [
                        html.Summary(f"{ind} (last: {ind_ts})"),
                        html.Ul(ticker_items, style={"margin-left": "1em"})
                    ],
                    open=ind_open,
                    className="mb-2"
                )
            )

        nodes.append(
            html.Details(
                [
                    html.Summary(f"{sector} (last: {sector_ts})"),
                    html.Div(ind_nodes, style={"margin-left": "1em"})
                ],
                open=sector_open,
                className="mb-3"
            )
        )

    return nodes

# ── Dash App Layout ────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Button("Play",    id="play-btn",    className="btn btn-primary"),
            html.Button("Pause",   id="pause-btn",   className="btn btn-warning ms-2", disabled=True),
            html.Button("Refresh", id="refresh-btn", className="btn btn-success ms-2"),
            html.Hr(),
            dcc.Input(id="ticker-search", type="text", placeholder="Search ticker...", debounce=True),
            html.Hr(),
            dcc.Interval(id="poller", interval=1000, n_intervals=0)
        ], width=3),
        dbc.Col(html.Div(id="tree-container"), width=6),
        dbc.Col(dcc.Textarea(id="log-area", style={"width":"100%","height":"80vh"}), width=3),
    ]),
], fluid=True)

# ── Poller & Search: Update tree & log pane ────────────────────────────────────
@app.callback(
    Output("tree-container", "children"),
    Output("log-area", "value"),
    Input("poller", "n_intervals"),
    Input("ticker-search", "value"),
)
def refresh(n_intervals, search_value):
    tree = build_tree(search_value)
    try:
        lines = PROGRESS_FILE.read_text().splitlines()[-LOG_LINES:]
        log   = "\n".join(lines)
    except Exception:
        log = ""
    return tree, log

# ── Play / Pause / Refresh Controls ────────────────────────────────────────────
@app.callback(
    Output("play-btn",    "disabled"),
    Output("pause-btn",   "disabled"),
    Output("refresh-btn", "disabled"),
    Input("play-btn",    "n_clicks"),
    Input("pause-btn",   "n_clicks"),
    Input("refresh-btn", "n_clicks"),
)
def control_ingest(play_ct, pause_ct, refresh_ct):
    global ingest_thread
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    btn = ctx.triggered[0]["prop_id"].split(".")[0]

    if btn == "play-btn":
        update_option_chains_in_db.STOP_EVENT.clear()
        if ingest_thread is None or not ingest_thread.is_alive():
            ingest_thread = threading.Thread(
                target=lambda: update_option_chains_in_db.main(full_run=False),
                daemon=True
            )
            ingest_thread.start()
        return True, False, False

    if btn == "pause-btn":
        update_option_chains_in_db.STOP_EVENT.set()
        return False, True, False

    if btn == "refresh-btn":
        update_option_chains_in_db.clear_progress()  # clear immediately
        update_option_chains_in_db.STOP_EVENT.clear()
        if ingest_thread is None or not ingest_thread.is_alive():
            ingest_thread = threading.Thread(
                target=lambda: update_option_chains_in_db.main(full_run=True),
                daemon=True
            )
            ingest_thread.start()
        return True, False, True

    raise PreventUpdate

# ── Run Server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
