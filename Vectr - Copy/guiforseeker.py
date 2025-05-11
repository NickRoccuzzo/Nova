# guiforseeker.py

from plotly.subplots import make_subplots
from dash import dcc, html
from datetime import datetime, timedelta
from extensions import cache
import yfinance as yf
import re
import logging
import time
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils import load_seeker_db
from flask import current_app
import math
from dash.dependencies import Input, Output, State

SETUPS: dict[str, list[tuple[str, float]]] = {"bullish": [], "bearish": []}
SECTORS: list[tuple[str, float]] = []

# ) Global constants
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
EMOJI_MAP = {
    "Small Whale":  "🐟",
    "Medium Whale": "🐬",
    "Large Whale":  "🦈",
    "Mega Whale":   "👑"
}
dash_theme = dbc.themes.BOOTSTRAP


# ) --- Cache & DB loader
@cache.memoize()
def _get_full_db():
    """
    Load the full JSON, then strip out only the ticker entries.
    If you've wrapped tickers under "tickers", pull that; otherwise,
    filter out the precomputed keys.
    """
    full = load_seeker_db(current_app.config["LOCAL_DB_FILE"])
    # Case A: you wrapped everything under "tickers"
    if isinstance(full, dict) and "tickers" in full:
        return full["tickers"]
    # Case B: flat JSON with extra keys "setups" & "sectors"
    return {
        key: val
        for key, val in full.items()
        if key not in ("setups", "sectors")
    }

# ) At top of guiforseeker.py
@cache.memoize()
def _get_full_db_raw() -> dict:
    # loads the entire seeker_database.json — cached in memory
    return load_seeker_db(current_app.config["LOCAL_DB_FILE"])

# ) Rewrite your two getters to use _get_full_db_raw()
@cache.memoize()
def get_precomputed_setups(direction: str) -> list[tuple[str, float]]:
    block = _get_full_db_raw().get("setups", {}).get(direction, [])
    return [(it["ticker"], it["score"]) for it in block]

@cache.memoize()
def get_precomputed_sectors() -> list[tuple[str, float]]:
    return [(it["sector"], it["score"]) for it in _get_full_db_raw().get("sectors", [])]


@cache.memoize()
def get_ticker_list() -> list[str]:
    """
    Return the list of all tickers (keys) from the filtered DB,
    so it excludes the precomputed 'setups' and 'sectors' blocks.
    """
    return list(_get_full_db().keys())


@cache.memoize()
def get_db_last_updated() -> datetime | None:
    """
    Returns the most recent timestamp across all tickers in the DB.
    """
    db = _get_full_db()
    times = []
    for meta in db.values():
        ts = meta.get("timestamp")
        if ts:
            try:
                times.append(datetime.fromisoformat(ts))
            except:
                pass
    return max(times) if times else None


# ) --- Normalization & parsing helpers
def _normalize_chain(entry: dict) -> dict:
    """
    Return the CALL/PUT chain in a stable shape:

        {
          "05/16/25": {
              "CALLS": [...],   # always present, maybe empty list
              "PUTS":  [...]
          },
          ...
        }

    Works with both legacy  {"data": {...}}
    and the new            {"by_volume": {...}}  layout.
    """
    # 1) pick whichever root key exists
    chain = entry.get("data") or entry.get("by_volume") or {}

    # 2) ensure each expiry has both side keys so later code never KeyErrors
    for exp, sides in chain.items():
        sides.setdefault("CALLS", [])
        sides.setdefault("PUTS",  [])

    return chain

@cache.memoize()
def get_ticker_meta(ticker: str) -> dict:
    """
    Returns:
      - timestamp: str of ISO timestamp when this ticker was last queried
      - expirations: int number of different expiration dates
    """
    db = _get_full_db()
    meta = db.get(ticker, {})
    ts = meta.get("timestamp", "")
    chain = _normalize_chain(meta.get("data", {}))
    return {
        "timestamp": ts,
        "expirations": len(chain)
    }

def parse_dollar_amount(s):
    """Parse '$1.4M' or '$750K' into float."""
    if not s:
        return 0.0
    s = s.replace("$", "").strip()
    mul = 1.0
    if s.endswith("M"):
        mul = 1e6; s = s[:-1]
    elif s.endswith("K"):
        mul = 1e3; s = s[:-1]
    try:
        return float(s) * mul
    except:
        return 0.0


def format_money(amount):
    """Compact formatting: >=1M → 'X.XM', else integer commas."""
    try:
        amt = float(amount)
    except:
        return "$0"
    return f"${amt/1e6:.1f}M" if amt >= 1e6 else f"${amt:,.0f}"


def is_recent_trade(date_str, max_hours=48):
    """True if lastTradeDate within max_hours."""
    try:
        parts = date_str.split()
        if parts and parts[-1].isalpha():
            parts = parts[:-1]
        clean = " ".join(parts)
        dt = datetime.strptime(clean, "%m/%d/%Y %I:%M:%S %p")
        return (datetime.now() - dt) <= timedelta(hours=max_hours)
    except:
        return False


def extract_strike_from_hover(text):
    """Pull the numeric strike from hover-text 'Strike: $XXX'."""
    for part in text.split("<br>"):
        if part.startswith("Strike:"):
            try:
                return float(part.split("Strike:")[1].strip().replace("$", ""))
            except:
                return np.nan
    return np.nan


# ) --- Unusual‑volumes extractor
def _extract_unusual_vols(chain, side="BOTH"):
    """
    Walk the normalized chain and pull out all volumes
    marked unusual (and recent) for CALLS, PUTS, or BOTH.
    """
    vols = []
    for contracts in chain.values():
        if side in ("CALL", "BOTH"):
            for c in contracts["CALLS"]:
                if c.get("unusual") and is_recent_trade(c.get("lastTradeDate","")):
                    try:
                        v = float(c.get("volume", 0))
                        if not math.isnan(v):
                            vols.append(v)
                    except:
                        pass
        if side in ("PUT", "BOTH"):
            for p in contracts["PUTS"]:
                if p.get("unusual") and is_recent_trade(p.get("lastTradeDate","")):
                    try:
                        v = float(p.get("volume", 0))
                        if not math.isnan(v):
                            vols.append(v)
                    except:
                        pass
    return vols

@cache.memoize()
def get_ticker_stats(ticker: str) -> dict:
    """
    Returns:
      - std: float   (std dev of recent unusual volumes)
      - count: int   (# of recent unusual trades)
      - max_spent: float  (largest total_spent)
    """
    db   = _get_full_db()
    meta = db.get(ticker)
    if not meta:
        return {"std":0.0, "count":0, "max_spent":0.0}

    chain = _normalize_chain(meta.get("data", {}))
    # pull only recent & unusual contracts
    unusual = []
    for day in chain.values():
        for side in ("CALLS","PUTS"):
            for c in day.get(side, []):
                if c.get("unusual") and is_recent_trade(c.get("lastTradeDate","")):
                    unusual.append(c)

    vols = [float(c.get("volume",0)) for c in unusual if c.get("volume",0) is not None]
    spend = [parse_dollar_amount(c.get("total_spent","")) for c in unusual]

    return {
        "std": float(np.std(vols)) if vols else 0.0,
        "count": len(unusual),
        "max_spent": max(spend) if spend else 0.0
    }


@cache.memoize()
def get_plot_data_for_ticker(ticker: str):
    """
    Returns the plot‑ready dict for a single ticker.
    """
    db   = _get_full_db()
    data = db.get(ticker)
    if not data:
        return None

    chain = _normalize_chain(data)
    return _build_plot_dict(chain)


def _build_plot_dict(chain):
    """
    Given a normalized chain (exp_date→{"CALLS":…,"PUTS":…}),
    produce the same pdict your old get_plot_data_for_ticker built.
    """
    from datetime import datetime

    # 1) sort expiration dates
    exp_dates = sorted(
        chain.keys(),
        key=lambda d: datetime.strptime(d, "%m/%d/%y")
    )

    pdict = {"exp_dates": exp_dates}

    # 2) for each side and rank, collect vols and texts
    for side in ("call", "put"):
        for r in (1, 2, 3):
            vols, texts = [], []
            for ed in exp_dates:
                lst = chain.get(ed, {}).get(side.upper() + "S", [])
                if len(lst) >= r:
                    c = lst[r - 1]
                    vols.append(c.get("volume", 0))
                    texts.append(
                        f"Strike: ${c.get('strike','')}<br>"
                        f"Total Spent: {c.get('total_spent','')}<br>"
                        f"Unusual: {c.get('unusual',False)}"
                    )
                else:
                    vols.append(0)
                    texts.append("N/A")
            pdict[f"{side}{r}_vol"]  = vols
            pdict[f"{side}{r}_text"] = texts

    return pdict


# 10) Layout builders

# ─── 6) LAYOUT BUILDERS ─────────────────────────────────────
def create_dash_layout():
    return html.Div([
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content")
    ])

def render_home_page():
    """
    Home page layout:
        • Primary‑mode radio (Daily Flow / Setups / Sectors)
        • Context panel (changes depending on primary mode)
        • Search box
        • Cards (rolodex) container
    """
    cfg         = current_app.config
    placeholder = cfg["SEARCH_PLACEHOLDER"]

    # ----- header row with all interactive controls -----------
    header_row = dbc.Row(
        [
            # PRIMARY MODE  (always visible)
            dbc.Col(
                dcc.RadioItems(
                    id="primary-mode",
                    options=[
                        {"label": "Daily Flow", "value": "daily"},
                        {"label": "Setups",     "value": "setups"},
                        {"label": "Sectors",    "value": "sectors"},
                    ],
                    value="daily",
                    inline=True,
                ),
                width="auto",
            ),

            # CONTEXT PANEL (contents supplied by a callback)
            dbc.Col(id="context-panel", width="auto"),

            # SEARCH BOX
            dbc.Col(
                dbc.Input(
                    id="ticker-search",
                    placeholder=placeholder,
                    type="text",
                    className="mb-0",
                ),
                width=4,
            ),
        ],
        className="mb-4 justify-content-center",
    )

    placeholders = html.Div(
        [
            dcc.Store(id="side-filter", data=None),
            dcc.Store(id="oi-direction", data=None),
        ],
        style={"display": "none"},
    )

    # ----- full page container --------------------------------
    return dbc.Container(
        [
            dbc.Row(dbc.Col(html.H2("Ticker Explorer",
                                    className="text-center my-4"))),
            header_row,
            placeholders,  # ← add this line
            dcc.Interval(
                id="interval-component",
                interval=30 * 1000,
                n_intervals=0,
            ),
            html.Div(id="cards-or-map"),

            dbc.Collapse(
                id="industry-panel",
                is_open=False,
                children=[],
                style={"marginTop": "2rem"},
            ),
        ],
        fluid=True,
    )

def ticker_detail_page(ticker):
    return html.Div([
        html.H2(f"Options for {ticker}", style={"textAlign":"center","marginTop":"2rem"}),
        dcc.RadioItems(
            id="price-timeframe-radio",
            options=[
                {"label":"1d","value":"1d"},
                {"label":"5d","value":"5d"},
                {"label":"1mo","value":"1mo"},
                {"label":"3mo","value":"3mo"},
                {"label":"6mo","value":"6mo"},
                {"label":"1y","value":"1y"},
                {"label":"5y","value":"5y"},
            ],
            value="3mo",
            inline=True,
            style={"textAlign":"center","marginBottom":"1rem"}
        ),
        dcc.Graph(id="ticker-detail-graph"),
        dcc.Interval(id="interval-component-detail", interval=30*1000, n_intervals=0)
    ])

# ─── Daily Flow ranking ─────────────────────────────────
from datetime import datetime
import numpy as np
from zoneinfo import ZoneInfo

BOOST_ALPHA = 1.0
TZ_ET       = ZoneInfo("America/New_York")

@cache.memoize()
def compute_ticker_ranking_std_side(
    tickers_tuple: tuple[str, ...],
    side: str = "BOTH",
) -> list[tuple[str, float]]:
    db = _get_full_db()

    def _days_out(exp_str: str) -> int:
        exp_dt = datetime.strptime(exp_str, "%m/%d/%y").replace(tzinfo=TZ_ET)
        return max((exp_dt - datetime.now(TZ_ET)).days, 0)

    # find farthest expiry
    all_days = [
        _days_out(exp)
        for entry in db.values()
        for exp in _normalize_chain(entry).keys()
    ]
    max_days_out = max([d for d in all_days if d > 0], default=1)

    ranking: dict[str, float] = {}
    for t in tickers_tuple:
        entry = db.get(t)
        if not entry:
            ranking[t] = 0.0
            continue

        weighted = []
        chain = _normalize_chain(entry)
        for exp, sides in chain.items():
            boost = 1 + BOOST_ALPHA * (_days_out(exp) / max_days_out)
            if side in ("CALL","BOTH"):
                for c in sides["CALLS"]:
                    if c.get("unusual") and is_recent_trade(c.get("lastTradeDate","")):
                        weighted.append(float(c.get("volume",0)) * boost)
            if side in ("PUT","BOTH"):
                for p in sides["PUTS"]:
                    if p.get("unusual") and is_recent_trade(p.get("lastTradeDate","")):
                        weighted.append(float(p.get("volume",0)) * boost)

        ranking[t] = float(np.std(weighted)) if weighted else 0.0

    return sorted(ranking.items(), key=lambda x: x[1], reverse=True)

# ─── ) Callbacks ────────────────────────────────────────────────────
def register_dash_callbacks(dash_app):
    @dash_app.callback(
        Output("page-content", "children"),
        Input("url", "pathname")
    )
    def display_page(path):
        if path in ("/", "/seeker-gui/"):
            return render_home_page()
        if path.startswith("/seeker-gui/ticker/"):
            ticker = path.split("/seeker-gui/ticker/")[1]
            return ticker_detail_page(ticker)
        return html.H3("404 Page Not Found", style={"textAlign":"center","marginTop":"2rem"})

    @dash_app.callback(
        Output("context-panel", "children"),
        Input("primary-mode", "value"),
    )
    def render_subcontrols(mode):
        if mode == "daily":
            return dcc.RadioItems(
                id="side-filter-radio",
                options=[
                    {"label":"Both","value":"BOTH"},
                    {"label":"Calls","value":"CALL"},
                    {"label":"Puts","value":"PUT"},
                ],
                value="BOTH",
                inline=True,
            )
        elif mode == "setups":
            return dcc.RadioItems(
                id="oi-direction-radio",
                options=[
                    {"label":"Bullish","value":"oi_above"},
                    {"label":"Bearish","value":"oi_below"},
                ],
                value="oi_above",
                inline=True,
            )
        # no extra controls for sectors now
        return None

    @dash_app.callback(
        Output("side-filter", "data"),
        Input("side-filter-radio", "value"),
        prevent_initial_call=True,
    )
    def save_side_filter(value):
        return value

    @dash_app.callback(
        Output("oi-direction", "data"),
        Input("oi-direction-radio", "value"),
        prevent_initial_call=True,
    )
    def save_oi_direction(value):
        return value

    @dash_app.callback(
        Output("cards-or-map", "children"),

        Input("primary-mode", "value"),
        Input("side-filter", "data"),  # ← store, always present
        Input("oi-direction", "data"),  # ← store, always present
        Input("ticker-search", "value"),
    )
    def update_main_view(mode, side_filter, oi_direction, search):
        """
        mode:            'daily' | 'setups' | 'sectors'
        side_filter:     'BOTH'|'CALL'|'PUT' for daily
        oi_direction:    'oi_above'|'oi_below' for setups
        search:          ticker search string
        """
        cfg = current_app.config
        max_cards = cfg["MAX_RANKING_CARDS"]

        # ─── DAILY ───────────────────────────────────────────────────────
        if mode == "daily":
            ranks = compute_ticker_ranking_std_side(
                tuple(get_ticker_list()),
                side_filter or "BOTH"
            )
            label = "Std Dev"

        # ─── SETUPS ─────────────────────────────────────────────────────
        elif mode == "setups":
            direction = "bullish" if (oi_direction or "oi_above") == "oi_above" else "bearish"
            ranks = get_precomputed_setups(direction)
            label = "OI ↑" if direction == "bullish" else "OI ↓"

        # ─── SECTORS ────────────────────────────────────────────────────
        else:
            scores = get_precomputed_sectors()
            scores = [(s, c) for s, c in scores if c > 0]
            if not scores:
                return html.Em("No sector data.")

            df = pd.DataFrame({
                "sector": [s for s, _ in scores],
                "score": [c for _, c in scores]
            })
            fig = px.treemap(
                df,
                path=["sector"],
                values="score",
                color="score",
                color_continuous_scale="Blues"
            ).update_layout(margin={"t": 40, "l": 10, "r": 10, "b": 10})

            return dcc.Graph(id="sector-treemap", figure=fig, style={"height": "70vh"})

        # ─── SHARED SEARCH FILTER & CARD BUILD ─────────────────────────
        if search:
            s = search.strip().upper()
            ranks = [r for r in ranks if s in r[0]]
        ranks = ranks[:max_cards]

        cards = []
        for idx, (tk, score) in enumerate(ranks):
            cards.append(
                dcc.Link(
                    dbc.Card([
                        dbc.CardHeader(f"#{idx + 1}", className="rolodex-rank"),
                        dbc.CardBody([
                            html.H5(tk, className="card-title"),
                            html.P(f"{label}: {score:,.2f}", className="card-text small"),
                        ]),
                    ], className="rolodex-card mb-3 p-3"),
                    href=f"/seeker-gui/ticker/{tk}",
                    style={"textDecoration": "none", "width": "100%"}
                )
            )

        return html.Div(cards, className="rolodex-container")

    @dash_app.callback(
        Output("ticker-detail-graph", "figure"),
        Input("url", "pathname"),
        Input("price-timeframe-radio", "value"),
        Input("interval-component-detail", "n_intervals")
    )
    def update_ticker_detail(path, tf, _):
        if path.startswith("/seeker-gui/ticker/"):
            ticker = path.split("/seeker-gui/ticker/")[1]
            return build_combined_graph(ticker, tf)
        return dash.no_update


def build_combined_graph(ticker: str, tf: str) -> go.Figure:
    """Build the combined price + option chain figure for a ticker and timeframe."""
    # --- Historical Price (left y1) ---
    try:
        yf_t = yf.Ticker(ticker)
        hist = yf_t.history(period=tf)
        if hist.empty:
            price_fig = go.Figure()
        else:
            price_fig = go.Figure([go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                line=dict(color="#6eff0d"),
                showlegend=False
            )])
    except:
        price_fig = go.Figure()

    # --- Option Chain (strike-lines on y1, bars on y2) ---
    pdata = get_plot_data_for_ticker(ticker)
    option_fig = go.Figure()
    exp_dates = pdata.get("exp_dates", []) if pdata else []

    if pdata:
        # Top-3 bars for CALLS & PUTS
        bar_colors = {
            "call": ["lightgreen", "green", "darkgreen"],
            "put":  ["salmon",     "red",   "darkred"]
        }
        for side in ("call", "put"):
            for rank, color in enumerate(bar_colors[side], start=1):
                vols = pdata.get(f"{side}{rank}_vol", [0] * len(exp_dates))
                option_fig.add_trace(go.Bar(
                    x=exp_dates,
                    y=vols,
                    marker_color=color,
                    opacity=0.6,
                    showlegend=False
                ))

        # Strike markers
        call_strikes, call_vols = get_top_strike_for_side(pdata, "call")
        put_strikes, put_vols = get_top_strike_for_side(pdata, "put")
        all_vols = call_vols + put_vols
        global_max = max(all_vols) if all_vols else 1
        min_ms, max_ms = 5, 60

        call_sizes = [
            min_ms + (v / global_max) * (max_ms - min_ms)
            for v in call_vols if not (v is None or np.isnan(v))
        ]

        put_sizes = [
            min_ms + (v / global_max) * (max_ms - min_ms)
            for v in put_vols if not (v is None or np.isnan(v))
        ]

        option_fig.add_trace(go.Scatter(
            x=exp_dates,
            y=call_strikes,
            mode="markers+lines",
            marker=dict(size=call_sizes, color="green"),
            showlegend=False
        ))
        option_fig.add_trace(go.Scatter(
            x=exp_dates,
            y=put_strikes,
            mode="markers+lines",
            marker=dict(size=put_sizes, color="red"),
            showlegend=False
        ))

    # --- Build combined figure with two y-axes ---
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        column_widths=[0.4, 0.6],
        horizontal_spacing=0.005,
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )

    for tr in price_fig.data:
        fig.add_trace(tr, row=1, col=1)

    for tr in option_fig.data:
        if isinstance(tr, go.Bar):
            fig.add_trace(tr, row=1, col=2, secondary_y=True)
        else:
            fig.add_trace(tr, row=1, col=2, secondary_y=False)

    fig.update_layout(
        template="plotly_white",
        height=800,
        margin={"l":50, "r":20, "t":50, "b":50},
        paper_bgcolor="#0d0b0c",
        plot_bgcolor="#0d0b0c",
        showlegend=False
    )

    fig.update_yaxes(autorange=True, title_text="Price", row=1, col=1, secondary_y=False, tickfont=dict(color="white", size=14))
    fig.update_yaxes(visible=False, row=1, col=2, secondary_y=False)
    fig.update_yaxes(autorange=True, title_text="", row=1, col=2, secondary_y=True)
    fig.update_xaxes(title_text="", row=1, col=1, tickangle=45, tickfont=dict(color="white", size=13))
    fig.update_xaxes(title_text="", row=1, col=2, tickangle=45, tickfont=dict(color="white", size=13))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # --- Annotations (optional) ---
    if pdata:
        cands = get_candidate_contracts(pdata)
        top5 = sorted(cands, key=lambda x: x["total_spent"], reverse=True)[:5]

        offs, anns = {}, []
        for c in top5:
            key = (c["exp_date"], c["strike"])
            idx = offs.get(key, 0)
            offs[key] = idx + 1
            ax, ay = 40 + idx * 40, -40

            strike_color = "white"
            side_color = "green" if c["side"] == "call" else "red"
            premium_color = "white"
            class_color = get_whale_color(c["classification"])

            classification = c["classification"]
            emoji = EMOJI_MAP.get(classification, "")
            label = f"{emoji} {classification}" if emoji else classification

            anns.append({
                "x": c["exp_date"],
                "y": c["strike"],
                "xref": "x2",
                "yref": "y1",
                "text": (
                    f"<span style='color:{strike_color};'>${c['strike']:.0f}</span> "
                    f"<span style='color:{side_color};'>{c['side'].upper()}</span><br>"
                    f"<span style='color:{premium_color};'>Qty: {c['volume']}</span><br>"
                    f"<span style='color:{premium_color};'>{format_money(c['total_spent'])}</span><br>"
                    f"<span style='color:{class_color};'>{label}</span>"
                ),
                "showarrow": True,
                "arrowhead": 2,
                "ax": ax,
                "ay": ay,
                "align": "left",
                "font": {"color": "white", "size": 10},
                "bgcolor": "#34393b",
                "bordercolor": "grey",
                "borderwidth": 1
            })

        if anns:
            fig.update_layout(annotations=anns)

    # --- Title ---
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
        price = info.get("regularMarketPrice")
        title = f"{name} ({ticker}) – Current Price: ${price:,.2f}" if price else name
    except:
        title = ticker
    fig.update_layout(title=title)

    return fig


def get_candidate_contracts(pdata):
    """
    From the plot data (pdata) for a ticker, build a list of candidate contracts
    that have the unusual flag (or otherwise qualify for whale annotation).

    Returns a list of dicts with keys:
      - "exp_date", "strike", "volume", "total_spent",
        "classification", "side"
    """
    candidates = []
    if pdata is None:
        return candidates

    n = len(pdata["exp_dates"])
    for i in range(n):
        exp_date = pdata["exp_dates"][i]
        for side in ("call", "put"):
            for rank in range(1, 4):
                txt_list = pdata.get(f"{side}{rank}_text", [])
                vol_list = pdata.get(f"{side}{rank}_vol", [])
                if i >= len(txt_list) or i >= len(vol_list):
                    continue
                raw_text = txt_list[i]
                if "Unusual: True" not in raw_text:
                    continue

                strike_val = extract_strike_from_hover(raw_text)
                if np.isnan(strike_val):
                    continue

                try:
                    volume = int(vol_list[i])
                except:
                    volume = 0

                # parse total_spent
                total_part = raw_text.split("Total Spent:")[-1].split("<br>")[0].strip()
                total_spent = parse_dollar_amount(total_part)

                classification = get_whale_category(total_spent) or "Unusual Trade"
                color = get_whale_color(classification)

                candidates.append({
                    "exp_date":      exp_date,
                    "strike":        strike_val,
                    "volume":        volume,
                    "total_spent":   total_spent,
                    "classification":classification,
                    "side":          side
                })
    return candidates


def strip_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)


def get_whale_category(amount):
    """Small/Medium/Large/Mega Whale based on total_spent."""
    try:
        amt = float(amount)
    except:
        return None
    if amt >= 10e6:
        return "Mega Whale"
    if amt >= 5e6:
        return "Large Whale"
    if amt >= 2e6:
        return "Medium Whale"
    if amt >= 0.8e6:
        return "Small Whale"
    return None


def get_whale_color(category):
    """Map a whale‐category to its display color."""
    return {
        "Small Whale":  "#9acd32",   # light green
        "Medium Whale": "#ffd700",   # gold
        "Large Whale":  "#ff8c00",   # dark orange
        "Mega Whale":   "#ff4500"    # red‐orange
    }.get(category, "white")


def get_annotation_contracts(pdata):
    """
    From pdata, gather all 'unusual' contracts as dicts:
    {'exp_date', 'strike', 'volume', 'total_spent', 'classification', 'side'}.
    """
    cands = []
    if not pdata:
        return cands

    for i, ed in enumerate(pdata["exp_dates"]):
        for side in ("call","put"):
            for r in (1,2,3):
                txts = pdata.get(f"{side}{r}_text", [])
                vols = pdata.get(f"{side}{r}_vol", [])
                if i >= len(txts):
                    continue
                txt = txts[i]
                if "Unusual: True" not in txt:
                    continue
                strike = extract_strike_from_hover(txt)
                if np.isnan(strike):
                    continue
                vol = vols[i] if i < len(vols) else 0
                # parse total_spent
                part = txt.split("Total Spent:")[-1].split("<br>")[0].strip()
                ts = parse_dollar_amount(part)
                cls = get_whale_category(ts) or "Unusual Trade"
                cands.append({
                    "exp_date": ed,
                    "strike": strike,
                    "volume": vol,
                    "total_spent": ts,
                    "classification": cls,
                    "side": side
                })
    return cands


def scale_marker_size(vol, scale=0.02, min_size=5, max_size=60):
    """Scale marker by volume."""
    try:
        s = vol * scale
    except:
        s = min_size
    return max(min_size, min(s, max_size))


def get_top_strike_for_side(pdata, side):
    """
    For each expiration, pick the strike of the single contract (rank1-3)
    with the highest volume. Returns (strikes[], volumes[]).
    """
    strikes, vols = [], []
    exp = pdata.get("exp_dates", [])
    n = len(exp)
    for i in range(n):
        tmp_v, tmp_s = [], []
        for r in (1,2,3):
            vol = pdata.get(f"{side}{r}_vol", [0]*n)[i]
            txt = pdata.get(f"{side}{r}_text", ["N/A"]*n)[i]
            tmp_v.append(vol)
            tmp_s.append(extract_strike_from_hover(txt))
        idx = int(np.argmax(tmp_v))
        strikes.append(tmp_s[idx])
        vols.append(tmp_v[idx])
    return strikes, vols


