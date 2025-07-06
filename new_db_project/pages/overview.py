# pages/overview.py

import json

import dash
import pandas as pd
from dash import html, dcc, callback, Input, Output, ALL, callback_context
from sqlalchemy import create_engine

from db_config import POSTGRES_DB_URL

dash.register_page(__name__, path="/", name="Overview")

# ─── Load hierarchy & z-scores ───────────────────────────────────────────────
with open('tickers.json') as f:
    tickers_hier = json.load(f)

engine = create_engine(POSTGRES_DB_URL)
tk_df = pd.read_sql(
    "SELECT t.symbol AS ticker, tm.z_score "
    "FROM ticker_metrics_zscores tm JOIN tickers t USING(ticker_id)",
    engine
)
z_map = dict(zip(tk_df['ticker'], tk_df['z_score']))
max_z = max(abs(z) for z in z_map.values()) or 1.0

# ─── Build nested <details> tree ─────────────────────────────────────────────
def build_tree(data, filt=""):
    filt = filt.lower()
    sectors = []
    for sector, inds in data.items():
        sec_vals, ind_blocks = [], []
        for industry, tickers in inds.items():
            ind_vals, items = [], []
            for t in tickers:
                sym, name = t['symbol'], t.get('full_name', t['symbol'])
                z = z_map.get(sym, 0.0)
                if filt and not any(f in txt for f, txt in [
                    (filt, sector.lower()),
                    (filt, industry.lower()),
                    (filt, sym.lower()),
                    (filt, name.lower())
                ]):
                    continue
                ind_vals.append(z); sec_vals.append(z)
                z_norm = max(-1, min(1, z/max_z))
                hue    = 120 if z_norm>0 else 0
                light  = 80 - 40*abs(z_norm)
                bg     = f"hsl({hue},60%,{light:.0f}%)"
                items.append(
                  html.Li(html.Button(
                    f"{sym} — {name} (z={z:.2f})",
                    id={'type':'ticker-btn','index':sym},
                    n_clicks=0,
                    className="ticker-btn",
                    tabIndex=0,
                    style={
                      "background": bg,
                      "border":"none",
                      "padding":"4px 8px",
                      "textAlign":"left",
                      "width":"100%",
                      "outline":"none"
                    }
                  ))
                )
            if not items: continue
            avg_iz = sum(ind_vals)/len(ind_vals)
            iz_norm= max(-1, min(1, avg_iz/max_z))
            hue_i, light_i = (120, 80-40*abs(iz_norm)) if iz_norm>0 else (0, 80-40*abs(iz_norm))
            bg_i = f"hsl({hue_i},60%,{light_i:.0f}%)"
            ind_blocks.append(html.Details([
                html.Summary(
                  industry,
                  className="industry-summary",
                  tabIndex=0,
                  style={
                    'cursor':'pointer',
                    'fontWeight':'bold',
                    'background': bg_i,
                    'padding':'4px'
                  }
                ),
                html.Ul(items, style={'marginLeft':'1rem','listStyle':'none','padding':0})
            ], open=False))
        if not ind_blocks: continue
        avg_sz = sum(sec_vals)/len(sec_vals)
        sz_norm= max(-1, min(1, avg_sz/max_z))
        hue_s, light_s = (120, 80-40*abs(sz_norm)) if sz_norm>0 else (0, 80-40*abs(sz_norm))
        bg_s = f"hsl({hue_s},60%,{light_s:.0f}%)"
        sectors.append(html.Details([
            html.Summary(
              sector,
              className="sector-summary",
              tabIndex=0,
              style={
                'cursor':'pointer',
                'fontSize':'1.1rem',
                'fontWeight':'bold',
                'background': bg_s,
                'padding':'6px'
              }
            ),
            html.Div(ind_blocks, style={'marginLeft':'1rem'})
        ], open=False))
    return sectors

# ─── Layout ──────────────────────────────────────────────────────────────────
layout = html.Div(
    style={"padding":"2rem","maxWidth":"1000px","margin":"auto"},
    children=[
        html.H1("Sectors → Industries → Tickers", style={"textAlign":"center"}),
        dcc.Input(
            id="filter-input",
            placeholder="Filter tickers…",
            debounce=True,
            style={"width":"100%","margin":"1rem 0"}
        ),
        html.Div([
            html.Span("High ↓ Low", style={"marginRight":"1rem"}),
            html.Span(style={
                "display":"inline-block",
                "width":"200px","height":"12px",
                "background":"linear-gradient(to right, red, lightgray, green)"
            }),
            html.Span("Low ↑ High", style={"marginLeft":"1rem"})
        ], style={"marginBottom":"1.5rem","textAlign":"center"}),
        html.Div(id="tree-container"),
        dcc.Location(id="loc")
    ]
)

# ─── Callbacks ────────────────────────────────────────────────────────────────
@callback(
    Output("tree-container","children"),
    Input("filter-input","value")
)
def update_tree(filter_val):
    return build_tree(tickers_hier, filter_val or "")

@callback(
    Output("loc","href"),
    Input({'type':'ticker-btn','index':ALL},'n_clicks')
)
def on_ticker_click(n_list):
    if not any(n_list):
        return dash.no_update
    ctx = callback_context
    triggered = ctx.triggered[0]
    import json
    btn_id = json.loads(triggered["prop_id"].split(".")[0])
    sym    = btn_id.get("index")
    return f"/graph-builder?symbol={sym}"
