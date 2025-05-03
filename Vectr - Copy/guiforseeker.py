# guiforseeker.py
from utils import load_seeker_db, load_seeker_db_s3
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output
from datetime import datetime, timedelta
from flask import current_app
from extensions import cache
import re
import logging
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import math

# 1) --- Global constants
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
EMOJI_MAP = {
    "Small Whale":  "🐟",
    "Medium Whale": "🐬",
    "Large Whale":  "🦈",
    "Mega Whale":   "👑"
}
dash_theme = dbc.themes.BOOTSTRAP

# 2) --- Cache & DB loader
@cache.memoize()
def _get_full_db():
    """
    Load & parse the entire seeker_database.json once per TTL.
    """
    return load_seeker_db(current_app.config["LOCAL_DB_FILE"])


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


# 3) --- Normalization & parsing helpers
def _normalize_chain(data: dict | list) -> dict[str, dict[str, list[dict]]]:
    """
    Turn either {"data": {...}} or a list-of-entries into
    a dict mapping exp_date → {"CALLS":[…], "PUTS":[…]}.
    """
    if isinstance(data, dict) and "data" in data:
        return data["data"]

    chain = {}
    for c in data if isinstance(data, list) else []:
        ed = c.get("date")
        if not ed:
            continue
        key = ed
        side = c.get("type", "").upper() + "S"
        chain.setdefault(key, {"CALLS": [], "PUTS": []})[side].append(c)

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


# 4) --- Unusual‑volumes extractor
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


# 5) --- Stats & ranking (all memoized!)
@cache.memoize()
def compute_ticker_ranking_std_side(tickers_tuple: tuple[str, ...], side: str = "BOTH"):
    """
    tickers_tuple: tuple of tickers so it's hashable for cache keys
    side: "CALL", "PUT", or "BOTH"
    """
    db      = _get_full_db()
    ranking = {}

    for t in tickers_tuple:
        data = db.get(t)
        if not data:
            ranking[t] = 0.0
            continue

        chain      = _normalize_chain(data)
        vols       = _extract_unusual_vols(chain, side)
        ranking[t] = float(np.std(vols)) if vols else 0.0

    # Return a sorted list by descending volatility
    return sorted(ranking.items(), key=lambda x: x[1], reverse=True)


@cache.memoize()
def compute_sector_ranking_std_side(side: str = "BOTH"):
    """
    For each sector, compute the average std‑dev of unusual‑volume trades
    (calls, puts or both) across all its tickers, then sort descending.
    """
    db = _get_full_db()

    # 1) Build sector → [tickers]
    sector_map: dict[str, list[str]] = {}
    for ticker, meta in db.items():
        sector = meta.get("sector") or "Unknown"
        sector_map.setdefault(sector, []).append(ticker)

    # 2) Compute each sector’s aggregated score
    sector_scores: dict[str, float] = {}
    for sector, tickers in sector_map.items():
        per_ticker = compute_ticker_ranking_std_side(tuple(tickers), side)
        if per_ticker:
            avg_std = sum(std for _, std in per_ticker) / len(per_ticker)
        else:
            avg_std = 0.0
        sector_scores[sector] = avg_std

    # 3) Sort descending
    return sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)

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
    exp_dates = sorted(chain.keys(),
                       key=lambda d: datetime.strptime(d, "%m/%d/%Y"))

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
    cfg         = current_app.config
    placeholder = cfg["SEARCH_PLACEHOLDER"]

    return dbc.Container([
        # Header & Controls
        dbc.Row(dbc.Col(html.H2("Ticker Explorer", className="text-center my-4"))),

        dbc.Row([
            # View switch
            dbc.Col(dcc.RadioItems(
                id="view-switch",
                options=[
                    {"label":"Master Ranking", "value":"master"},
                    {"label":"Sector View",    "value":"sector"},
                ],
                value="master",
                inline=True
            ), width="auto"),

            # Call/Put/Both
            dbc.Col(dcc.RadioItems(
                id="ranking-type-radio",
                options=[
                    {"label":"Both",       "value":"BOTH"},
                    {"label":"Calls Only", "value":"CALL"},
                    {"label":"Puts Only",  "value":"PUT"},
                ],
                value="BOTH",
                inline=True
            ), width="auto"),

            # Search box
            dbc.Col(dbc.Input(
                id="ticker-search",
                placeholder=placeholder,
                type="text",
                className="mb-0"
            ), width=4),
        ], className="mb-4 justify-content-center"),

        # Interval (for auto‑refresh)
        dcc.Interval(id="interval-component", interval=30*1000, n_intervals=0),

        # Cards container
        html.Div(id="cards-container", className="card-stack")
    ], fluid=True)

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


# 11) Callbacks
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
        Output("cards-container", "children"),
        Input("view-switch", "value"),
        Input("ranking-type-radio", "value"),
        Input("ticker-search", "value"),
        Input("interval-component", "n_intervals"),
    )
    def update_cards(view, side, search, _):
        cfg        = current_app.config
        max_cards  = cfg["MAX_RANKING_CARDS"]
        cols       = cfg["RANKING_COLUMNS"]
        delay_ms   = cfg["CARD_ANIMATION_DELAY_MS"]
        col_width  = max(1, 12 // cols)

        # ─── Master Ranking ─────────────────────
        if view == "master":
            # 1) compute per‑ticker std dev
            ranks = compute_ticker_ranking_std_side(
                tuple(get_ticker_list()), side=side
            )
            # 2) apply search filter + truncate
            if search:
                s = search.strip().upper()
                ranks = [r for r in ranks if s in r[0]]
            ranks = ranks[:max_cards]

            # 3) build a rolodex‑style vertical list
            cards = []
            for i, (ticker, score) in enumerate(ranks):
                cards.append(
                    dcc.Link(
                        dbc.Card([
                            # big, left‑aligned rank header
                            dbc.CardHeader(
                                f"#{i + 1}",
                                className="rolodex-rank"
                            ),
                            dbc.CardBody([
                                html.H5(ticker, className="card-title"),
                                html.P(f"Std Dev: {score:.2f}", className="card-text small"),
                            ])
                        ],
                            className="rolodex-card mb-3 p-3",
                        ),
                        href=f"/seeker-gui/ticker/{ticker}",
                        style={"textDecoration": "none", "width": "100%"}
                    )
                )

            return html.Div(
                cards,
                className="rolodex-container"
            )

        # ─── Sector View with drill‑in ───────────────────
        else:
            from dash_bootstrap_components import Accordion, AccordionItem

            # 1) pull full DB
            db = _get_full_db()

            # 2) group tickers by sector
            sector_groups = {}
            for tk, meta in db.items():
                sec = meta.get("sector", "Unknown")
                sector_groups.setdefault(sec, []).append(tk)

            # 3) compute aggregate score per sector
            sector_scores = []
            for sec, tickers in sector_groups.items():
                stds = [
                    compute_ticker_ranking_std_side((tk,), side=side)[0][1]
                    for tk in tickers
                ]
                agg = float(np.sum(stds)) if stds else 0.0
                sector_scores.append((sec, agg, tickers))

            # 4) sort & limit
            sector_scores.sort(key=lambda x: x[1], reverse=True)
            sector_scores = sector_scores[:max_cards]

            # 5) build an accordion: one panel per sector
            items = []
            for sec, agg, members in sector_scores:
                # group this sector’s tickers by industry
                industry_groups = {}
                for tk in members:
                    ind = db[tk].get("industry", "Unknown")
                    industry_groups.setdefault(ind, []).append(tk)

                # build the inside of the accordion for this sector
                children = []
                for ind, tks in industry_groups.items():
                    children.append(html.H6(ind, style={"marginTop": "1rem"}))
                    children.append(
                        html.Ul(
                            [html.Li(t) for t in sorted(tks)],
                            style={"marginLeft": "1rem"}
                        )
                    )

                items.append(
                    AccordionItem(
                        title=f"{sec} — Aggregate Std: {agg:.2f}",
                        children=children
                    )
                )

            return html.Div(
                Accordion(
                    items,
                    start_collapsed=True,
                    always_open=True,
                    flush=True
                ),
                style={"marginTop": "1rem"}
            )

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

def get_ticker_list():
    seeker_db = load_seeker_db()
    return list(seeker_db.keys())

def load_json_for_ticker(ticker):
    seeker_db = load_seeker_db()
    return seeker_db.get(ticker)

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


