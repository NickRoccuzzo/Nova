import os
import logging
from functools import lru_cache

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_session import Session
from flask_wtf import CSRFProtect
from werkzeug.exceptions import BadRequest
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import yfinance as yf
import socket
import json
import shutil
from plotly.utils import PlotlyJSONEncoder
from time import time
from VectrPyLogic import save_options_data, calculate_and_visualize_data
from sectors import get_etf_performance

# --- Import your seeker database function ---
from guiforseeker import create_dash_layout, register_dash_callbacks, dash_theme, load_seeker_db_s3

SYNC_TTL = 60 * 60  # seconds

# Configuration
class Config:
    SECRET_KEY         = os.getenv("FLASK_SECRET_KEY", "replace-this-with-a-random-secret")
    SESSION_TYPE       = "filesystem"
    SESSION_FILE_DIR   = os.getenv("SESSION_FILE_DIR", "/tmp/flask_session")
    SESSION_PERMANENT  = False
    SESSION_USE_SIGNER = True
    S3_BUCKET          = os.getenv("S3_BUCKET", "nova-93")
    SP500_ETFS         = ["XLRE","XLE","XLU","XLK","XLB","XLP","XLY","XLI","XLC","XLV","XLF","XBI"]
    DEBUG              = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# App and extensions initialization
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.before_request
def auto_sync_seeker_db():
    # only for Contact or Dash routes
    if not (request.path.startswith("/seeker-gui") or request.path == "/contact"):
        return

    # AWS creds check
    ak = session.get("aws_access_key_id")
    sk = session.get("aws_secret_access_key")
    if not ak or not sk:
        return

    now = time()
    last = session.get("seeker_last_sync", 0)
    if now - last < SYNC_TTL:
        # Too soon—skip talking to S3
        return

    local_path = os.path.join(os.getcwd(), "seeker_database.json")
    try:
        load_seeker_db_s3(
            s3_uri=f"s3://{app.config['S3_BUCKET']}/seeker_database.json",
            aws_key=ak,
            aws_secret=sk,
            local_path=local_path
        )
        # record we just checked
        session["seeker_last_sync"] = now
    except Exception as e:
        logging.debug("Auto-sync seeker DB failed: %s", e)


# 👇 INSERT HERE
@app.before_request
def disable_csrf_for_dash():
    if request.path.startswith("/seeker-gui/"):
        setattr(request, "_disable_csrf", True)

# Now load configs
app.config.from_object(Config)
Session(app)


# Security headers
@app.after_request
def apply_security_headers(response):
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

# Helper functions
@lru_cache(maxsize=128)
def get_daily_change(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        return info.get("regularMarketChangePercent")
    except Exception as e:
        logging.warning("Error fetching market change for %s: %s", ticker, e)
        return None

def load_custom_etfs():
    path = os.path.join(app.root_path, "templates", "custom_etfs.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logging.error("Failed to load custom_etfs.json: %s", e)
        return {}

def get_custom_etf_data(name, tickers):
    rows = []
    for t in tickers:
        change = get_daily_change(t)
        color = "green" if (change or 0) > 0 else "red"
        change_str = f'<span style="color:{color};">{round(change,2) if change is not None else "N/A"}%</span>'
        link = f'<a href="/?ticker={t}" target="_self">{t}</a>'
        rows.append({"Ticker": link, "Daily Change (%)": change_str})
    df = pd.DataFrame(rows)
    return df.to_html(classes="table table-striped", index=False, escape=False)

def _load_holdings_for(etf):
    file = os.path.join("sectors", f"{etf}_holdings.xlsx")
    if not os.path.exists(file):
        return None, '<p>Holdings data being updated.</p>'
    try:
        df = pd.read_excel(file)
        from holdings import enhance_holdings_data
        df = enhance_holdings_data(df)
        if "Weight" in df.columns:
            df["Weight"] = df["Weight"].apply(lambda x: f"{x}%" if pd.notnull(x) else "N/A")
        html = df.to_html(classes="table table-striped holdings-table", index=False, escape=False)
        perf = get_etf_performance([etf]).get(etf, {})
        return perf, html
    except Exception as e:
        logging.error("Error loading holdings for %s: %s", etf, e)
        return None, '<p>Error loading holdings data.</p>'

def _load_sector_data(etfs):
    perf = get_etf_performance(etfs)
    holdings = {}
    for e in etfs:
        _, html = _load_holdings_for(e)
        holdings[e] = html
    return perf, holdings

# Routes
@app.route("/", methods=["GET"])
def index():
    aws_connected = "aws_access_key_id" in session and "aws_secret_access_key" in session
    local_path     = os.path.join(os.getcwd(), "seeker_database.json")
    seeker_missing = not os.path.exists(local_path)

    return render_template(
        "index.html",
        aws_connected=aws_connected,
        seeker_missing=seeker_missing
    )

# --- CONTACT ROUTE: auto-download if missing, redirect to Dash ---
@app.route("/contact")
def contact():
    local_path = os.path.join(os.getcwd(), "seeker_database.json")

    # if the file isn't already in cwd, try to pull it down
    if not os.path.exists(local_path):
        # require AWS creds
        if "aws_access_key_id" not in session or "aws_secret_access_key" not in session:
            flash("AWS credentials missing. Please login first.", "danger")
            return redirect(url_for("index"))

        # download from S3 into cwd
        try:
            load_seeker_db_s3(
                s3_uri=f"s3://{app.config['S3_BUCKET']}/seeker_database.json",
                aws_key=session["aws_access_key_id"],
                aws_secret=session["aws_secret_access_key"],
                local_path=local_path
            )

        except Exception as e:
            flash(f"Failed to download Seeker database: {e}", "danger")
            return redirect(url_for("index"))

    # whether we just downloaded it or it was already there, go to Dash
    return redirect("/seeker-gui/")


@app.route("/sectors")
def sectors():
    perf, holdings = _load_sector_data(app.config["SP500_ETFS"])
    custom = {n: get_custom_etf_data(n, t) for n, t in load_custom_etfs().items()}
    return render_template("SP500sectors.html", performance=perf, holdings_data=holdings, custom_etf_tables=custom)

@app.route("/get_performance")
def get_performance():
    etf = request.args.get("etf")
    timeframe = request.args.get("timeframe")
    if not etf or not timeframe:
        raise BadRequest("Missing parameters")
    perf = get_etf_performance([etf])
    return jsonify({"performance": perf.get(etf, {}).get(timeframe, "N/A")})

@app.route("/get_performance_group")
def get_performance_group():
    timeframe = request.args.get("timeframe")
    if not timeframe:
        raise BadRequest("Missing timeframe")
    perf = get_etf_performance(app.config["SP500_ETFS"])
    return jsonify({e: {"performance": perf.get(e, {}).get(timeframe)} for e in app.config["SP500_ETFS"]})

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ak = request.form.get("access_key", "").strip()
        sk = request.form.get("secret_key", "").strip()
        s3 = boto3.client(
            "s3",
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
        )

        # 1) Validate credentials by HEAD on the JSON
        try:
            s3.head_object(Bucket=app.config["S3_BUCKET"], Key="seeker_database.json")
        except ClientError:
            flash("Invalid AWS credentials or no access.", "danger")
        else:
            # 2) Store creds in session
            session["aws_access_key_id"]     = ak
            session["aws_secret_access_key"] = sk

            # 3) Immediately download the latest JSON
            local_path = os.path.join(os.getcwd(), "seeker_database.json")
            try:
                load_seeker_db_s3(
                    s3_uri=f"s3://{app.config['S3_BUCKET']}/seeker_database.json",
                    aws_key=ak,
                    aws_secret=sk,
                    local_path=local_path
                )
                flash("Seeker database downloaded successfully! ✅", "success")
            except Exception as e:
                # Warn but don’t block login
                flash(f"Warning: could not auto-download Seeker DB: {e}", "warning")

            # 4) Redirect back (or to index)
            return redirect(request.args.get("next") or url_for("index"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("aws_access_key_id", None)
    session.pop("aws_secret_access_key", None)
    return redirect(url_for('index'))

@app.route("/list_buckets")
def list_buckets():
    ak = session.get("aws_access_key_id")
    sk = session.get("aws_secret_access_key")
    if not ak or not sk:
        return redirect(url_for("login", next=url_for("list_buckets")))
    s3 = boto3.client("s3", aws_access_key_id=ak, aws_secret_access_key=sk)
    try:
        buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    except ClientError:
        flash("Error listing buckets.", "danger")
        buckets = []
    return render_template("buckets.html", buckets=buckets)

@app.route("/seeker-gui")
def seeker_gui():
    return redirect("http://127.0.0.1:8050/")

@app.context_processor
def dash_running():
    try:
        sock = socket.create_connection(("127.0.0.1", 8050), timeout=0.2)
        sock.close()
        return dict(dash_running=True)
    except OSError:
        return dict(dash_running=False)

@app.route("/getting-started")
def getting_started():
    return render_template("getting_started.html")

@app.route("/sp500sectors")
def sp500sectors_alias():
    return redirect(url_for("sectors"))

@app.route("/process_ticker", methods=["GET", "POST"])
def process_ticker():
    ticker = (
        request.args.get("ticker") if request.method == "GET"
        else request.form.get("ticker", "")
    ).strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400
    try:
        save_options_data(ticker)
        fig = calculate_and_visualize_data(ticker, width=1600, height=590)
        graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
        tmp = os.path.join(os.getcwd(), ticker)
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        return jsonify({"ticker": ticker, "graph_json": graph_json})
    except Exception as e:
        logging.exception("Error processing ticker %s", ticker)
        return jsonify({"error": str(e)}), 500

import dash
dash_app = dash.Dash(
    __name__,
    server=app,
    url_base_pathname='/seeker-gui/',
    external_stylesheets=[dash_theme]
)
dash_app.layout = create_dash_layout()
register_dash_callbacks(dash_app)


if __name__ == "__main__":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run(debug=True)

