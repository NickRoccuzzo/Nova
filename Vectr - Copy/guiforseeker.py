# guiforseeker.py

import re
import os
import json
import time
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import math
dash_theme = dbc.themes.BOOTSTRAP
# ——————————————————————————————————————————————
# Logging
# ——————————————————————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ——————————————————————————————————————————————
# Cache & S3 settings
# ——————————————————————————————————————————————
CACHE_TTL     = 3600  # seconds to cache the JSON locally
LOCAL_DB_FILE = "seeker_database.json"
S3_URI        = "s3://nova-93/seeker_database.json"


def load_seeker_db_s3(
    s3_uri: str,
    aws_key: str,
    aws_secret: str,
    local_path: str = "seeker_database.json"
) -> dict:
    """
    Load seeker_database.json from S3, downloading only if:
      • local_path doesn't exist, or
      • the remote S3 object has a newer LastModified timestamp than local_path.
    """
    # Parse bucket & key from s3:// URI
    p      = urlparse(s3_uri)
    bucket = p.netloc
    key    = p.path.lstrip("/")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
    )

    need_download = False

    if not os.path.exists(local_path):
        logging.info("Local DB not found—will download from S3.")
        need_download = True
    else:
        # Compare remote vs local
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            remote_mod = head["LastModified"]
            # local file mtime (as datetime, with same tzinfo)
            local_mtime = datetime.fromtimestamp(
                os.path.getmtime(local_path),
                tz=remote_mod.tzinfo
            )
            if remote_mod > local_mtime:
                logging.info("Remote DB is newer—will download updated version.")
                need_download = True
            else:
                logging.info("Local DB is up-to-date; skipping download.")
        except ClientError as e:
            logging.warning(f"S3 HEAD failed ({e}); using existing local copy if present.")
            # if HEAD errors, we leave need_download=False so we’ll try local

    if need_download:
        try:
            s3.download_file(bucket, key, local_path)
            logging.info(f"Downloaded seeker DB to {local_path}")
        except ClientError as e:
            logging.error(f"S3 download failed: {e}")
            # If we don’t have any local copy, we must fail
            if not os.path.exists(local_path):
                raise

    # Load and return JSON
    with open(local_path, "r") as f:
        return json.load(f)


logging.info("Seeker database loaded; ready to build your Dash app.")

### Emojis used for Annotations ###
EMOJI_MAP = {
    "Small Whale":  "🐟",
    "Medium Whale": "🐬",
    "Large Whale":  "🦈",
    "Mega Whale":   "👑"
}

# ---------------- Helper Functions ----------------

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


def format_money(amount):
    """Compact formatting: >=1M → 'X.XM', else integer commas."""
    try:
        amt = float(amount)
    except:
        return "$0"
    return f"${amt/1e6:.1f}M" if amt >= 1e6 else f"${amt:,.0f}"


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


def strip_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)


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


def scale_marker_size(vol, scale=0.02, min_size=5, max_size=60):
    """Scale marker by volume."""
    try:
        s = vol * scale
    except:
        s = min_size
    return max(min_size, min(s, max_size))


def extract_strike_from_hover(text):
    """Pull the numeric strike from hover-text 'Strike: $XXX'."""
    for part in text.split("<br>"):
        if part.startswith("Strike:"):
            try:
                return float(part.split("Strike:")[1].strip().replace("$", ""))
            except:
                return np.nan
    return np.nan


def get_ticker_list():
    """Load seeker_database.json and return list of tickers."""
    try:
        with open("seeker_database.json", "r") as f:
            seeker_db = json.load(f)
        return list(seeker_db.keys())
    except Exception as e:
        logging.error(f"Error loading seeker_database.json: {e}")
        return []


def load_json_for_ticker(ticker):
    """Load seeker_database.json and return data for a specific ticker."""
    try:
        with open("seeker_database.json", "r") as f:
            seeker_db = json.load(f)
        return seeker_db.get(ticker)
    except Exception as e:
        logging.error(f"Error loading seeker_database.json: {e}")
        return None



# ---------------- Data Processing ----------------


def compute_ticker_ranking_std_side(tickers, side="BOTH"):
    """
    For each ticker, gather all 'unusual' volumes in the last 48h,
    compute their std dev, and sort descending.
    """
    ranking = {}
    for t in tickers:
        try:
            data = load_json_for_ticker(t)
        except:
            continue

        # Normalize into date→{"CALLS":[..],"PUTS":[..]}
        chain = data.get("data") if isinstance(data, dict) and "data" in data else {}
        if not chain and isinstance(data, list):
            chain = {}
            for c in data:
                ed = c.get("date")
                if not ed:
                    continue
                chain.setdefault(ed, {"CALLS": [], "PUTS": []})
                chain[ed][c.get("type") + "S"].append(c)

        vols = []
        def add_vol(c):
            if c.get("unusual") and is_recent_trade(c.get("lastTradeDate","")):
                try:
                    v = float(c.get("volume",0))
                    if not math.isnan(v):
                        vols.append(v)
                except:
                    pass

        for contracts in chain.values():
            if side in ("CALL","BOTH"):
                for c in contracts.get("CALLS", []):
                    add_vol(c)
            if side in ("PUT","BOTH"):
                for p in contracts.get("PUTS", []):
                    add_vol(p)

        ranking[t] = float(np.std(vols)) if vols else 0.0

    return sorted(ranking.items(), key=lambda x: x[1], reverse=True)


def get_plot_data_for_ticker(ticker):
    """
    Reads the JSON, returns a dict with:
      'exp_dates': [dates...],
      'call1_vol', 'call1_text', ..., 'put3_vol','put3_text'
    """
    try:
        data = load_json_for_ticker(ticker)
    except:
        return None

    chain = data.get("data") if isinstance(data, dict) and "data" in data else {}
    if not chain and isinstance(data, list):
        chain = {}
        for c in data:
            ed = c.get("date")
            if not ed:
                continue
            chain.setdefault(ed, {"CALLS": [], "PUTS": []})
            chain[ed][c.get("type") + "S"].append(c)

    exp_dates = sorted(chain.keys(),
                       key=lambda d: datetime.strptime(d, "%m/%d/%Y"))
    pdict = {"exp_dates": exp_dates}
    for side in ("call","put"):
        for r in (1,2,3):
            vols, texts = [], []
            for ed in exp_dates:
                lst = chain.get(ed, {}).get(side.upper()+"S", [])
                if len(lst) >= r:
                    c = lst[r-1]
                    vols.append(c.get("volume", 0))
                    texts.append(
                        f"Strike: ${c.get('strike','')}<br>"
                        f"Total Spent: {c.get('total_spent','')}<br>"
                        f"Unusual: {c.get('unusual',False)}"
                    )
                else:
                    vols.append(0)
                    texts.append("N/A")
            pdict[f"{side}{r}_vol"] = vols
            pdict[f"{side}{r}_text"] = texts

    return pdict


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


# ---------------- App Layout ----------------

# ---------------- Dash App Layout Builder ----------------

def create_dash_layout():
    return html.Div([
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content")
    ])

# ---------------- Register Dash Callbacks ----------------

def render_ranking_page():
    ranks = compute_ticker_ranking_std_side(get_ticker_list(), side="BOTH")[:20]

    return dbc.Container([
        dbc.Row(dbc.Col(html.H2("Master Ticker Rankings", className="text-center my-4"))),

        dbc.Row(dbc.Col([
            html.Div([
                dbc.Input(id="ticker-search", placeholder="Search ticker...", type="text", className="mb-4"),
            ], className="text-center"),
        ], width=6)),

        dbc.Row([
            dbc.Col(
                dcc.Link([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(ticker, className="card-title"),
                            html.P(f"Std Dev: {value:.2f}", className="card-text small")
                        ])
                    ], className="mb-4 simple-card", style={"animationDelay": f"{i * 0.1:.1f}s"})
                ], href=f"/seeker-gui/ticker/{ticker}", style={"textDecoration": "none"}),
                width=4,
            )
            for i, (ticker, value) in enumerate(ranks)
        ])
    ], fluid=True)


def industry_graph_page():
    return html.Div([
        html.H2("Sector Rankings", style={"textAlign": "center", "marginTop": "2rem"}),
        dcc.Graph(id="industry-graph"),
        dcc.Interval(id="interval-component-industry", interval=30*1000, n_intervals=0)
    ])

def ticker_detail_page(ticker):
    return html.Div([
        html.H2(f"Options for {ticker}", style={"textAlign": "center", "marginTop": "2rem"}),
        dcc.RadioItems(
            id="price-timeframe-radio",
            options=[
                {"label": "1d", "value": "1d"},
                {"label": "5d", "value": "5d"},
                {"label": "1mo", "value": "1mo"},
                {"label": "3mo", "value": "3mo"},
                {"label": "6mo", "value": "6mo"},
                {"label": "1y", "value": "1y"},
                {"label": "5y", "value": "5y"}
            ],
            value="3mo",
            inline=True,
            style={"textAlign": "center", "marginBottom": "1rem"}
        ),
        dcc.Graph(id="ticker-detail-graph"),
        dcc.Interval(id="interval-component-detail", interval=30*1000, n_intervals=0)
    ])


def register_dash_callbacks(dash_app):
    @dash_app.callback(
        dash.dependencies.Output("page-content", "children"),
        dash.dependencies.Input("url", "pathname")
    )
    def display_page(path):
        if path in ("/", "/seeker-gui/"):
            return render_ranking_page()
        if path.startswith("/seeker-gui/industry"):
            return industry_graph_page()
        if path.startswith("/seeker-gui/ticker/"):
            ticker = path.split("/seeker-gui/ticker/")[1]
            return ticker_detail_page(ticker)
        return html.H3("404 Page Not Found", style={"textAlign": "center", "marginTop": "2rem"})

    @dash_app.callback(
        Output("ranking-list", "children"),
        Input("ranking-type-radio", "value"),
        Input("ticker-search", "value"),
        Input("interval-component", "n_intervals")
    )
    def update_ranking(rtype, search, _):
        ranks = compute_ticker_ranking_std_side(get_ticker_list(), side=rtype)
        if search:
            ranks = [r for r in ranks if search.upper() in r[0]]
        ranks = ranks[:20]

        cards = []
        for i, (ticker, score) in enumerate(ranks):
            cards.append(
                html.Div([
                    html.Div([
                        html.Div([
                            html.H5(f"#{i + 1}: {ticker}", className="card-title"),
                            html.P(f"Std Dev: {score:.2f}", className="card-text"),
                        ], className="card-front"),
                        html.Div([
                            html.H6("Top Whale Trade", className="card-title"),
                            html.P("Coming Soon!", className="card-text"),
                            dbc.Button("View", href=f"/seeker-gui/ticker/{ticker}", color="primary", size="sm")
                        ], className="card-back"),
                    ], className="flip-inner"),
                ], className="flip-card m-2", style={"animationDelay": f"{i * 0.1}s"})
            )
        return html.Div(
            cards,
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "marginTop": "2rem"
            }
        )

    @dash_app.callback(
        dash.dependencies.Output("industry-graph", "figure"),
        dash.dependencies.Input("interval-component-industry", "n_intervals")
    )
    def update_industry_graph(_):
        return industry_graph_page().children[1].children[0].figure

    @dash_app.callback(
        dash.dependencies.Output("ticker-detail-graph", "figure"),
        dash.dependencies.Input("url", "pathname"),
        dash.dependencies.Input("price-timeframe-radio", "value"),
        dash.dependencies.Input("interval-component-detail", "n_intervals")
    )
    def update_ticker_detail_combined(path, tf, _):
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