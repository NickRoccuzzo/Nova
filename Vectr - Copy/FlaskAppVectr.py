# Python Imports #
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from botocore.exceptions import ClientError
from plotly.utils import PlotlyJSONEncoder
from flask_session import Session
from extensions import cache
from functools import wraps
from time import time
import logging
import socket
import shutil
import boto3
import json
import dash
import re
import os

# --- Imports from Local Python Files: 'guiforseeker.py', 'VectrPyLogic.py', and 'utils.py'
from VectrPyLogic import save_options_data, calculate_and_visualize_data
from utils import download_seeker_db_if_missing, load_seeker_db_s3

### CONFIGURATION DASHBOARD ###
class Config:
    # ─────────── Security Settings ───────────
    SECRET_KEY        = os.getenv("FLASK_SECRET_KEY", "replace-this-with-a-random-secret")

    # ─────────── Session Settings ────────────
    SESSION_TYPE      = "filesystem"
    SESSION_FILE_DIR  = os.getenv("SESSION_FILE_DIR", "/tmp/flask_session")
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER= True

    # ─────────── S3 / Storage Settings ───────────
    S3_BUCKET         = os.getenv("S3_BUCKET",     "nova-93")
    S3_DB_URI         = os.getenv("S3_DB_URI",     "s3://nova-93/seeker_database.json")
    S3_DB_KEY         = os.getenv("S3_DB_KEY",     "seeker_database.json")
    LOCAL_DB_FILE     = os.getenv("LOCAL_DB_FILE", "seeker_database.json")

    # ──────── Caching & Sync Settings ─────────
    DB_CACHE_TTL      = int(os.getenv("DB_CACHE_TTL",      "3600"))  # seconds to keep local cache
    SYNC_TTL_SECONDS  = int(os.getenv("SYNC_TTL_SECONDS",  "3600"))  # seconds between auto-syncs

    # ───────── Validation Patterns ────────────
    TICKER_PATTERN    = r"^[A-Z0-9\.]{1,5}$"

    # ──────────── Debug Mode ──────────────────
    DEBUG             = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    # ─────── Ranking Page Settings ────────
    MAX_RANKING_CARDS = int(os.getenv("MAX_RANKING_CARDS", "30"))
    RANKING_COLUMNS = int(os.getenv("RANKING_COLUMNS", "3"))
    CARD_ANIMATION_DELAY_MS = int(os.getenv("CARD_ANIMATION_DELAY_MS", "100"))
    SEARCH_PLACEHOLDER = os.getenv("SEARCH_PLACEHOLDER", "Search ticker...")

    # ──────── Caching Backend & TTL ──────────
    CACHE_TYPE = os.getenv("CACHE_TYPE", "simple")
    CACHE_DEFAULT_TIMEOUT = int(os.getenv("CACHE_DEFAULT_TIMEOUT", DB_CACHE_TTL))


# LOG Level Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# App = Flask
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load Configs
app.config.from_object(Config)
TICKER_RE = re.compile(app.config["TICKER_PATTERN"])
Session(app)

cache.init_app(app, config={
    "CACHE_TYPE":            app.config["CACHE_TYPE"],
    "CACHE_DEFAULT_TIMEOUT": app.config["CACHE_DEFAULT_TIMEOUT"],
})

from guiforseeker import create_dash_layout, register_dash_callbacks, dash_theme, load_seeker_db_s3

@app.before_request
def preflight():
    if not request.path.startswith("/seeker-gui"):
        return

    # disable CSRF for Dash
    setattr(request, "_disable_csrf", True)

    # only auto‑sync if TTL expired
    last = session.get("seeker_last_sync", 0)
    if time() - last >= app.config["SYNC_TTL_SECONDS"]:
        if download_seeker_db_if_missing():
            session["seeker_last_sync"] = time()

# Security headers
@app.after_request
def apply_security_headers(response):
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

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

def ensure_seeker_db(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not download_seeker_db_if_missing():
            return redirect(url_for("index"))
        return func(*args, **kwargs)
    return wrapper

# --- CONTACT ROUTE: auto-download if missing, redirect to Dash ---
@app.route("/contact")
@ensure_seeker_db
def contact():
    return redirect("/seeker-gui/")

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

@app.route("/process_ticker", methods=["GET", "POST"])
def process_ticker():
    # 1) Extract and normalize
    raw = (request.args.get("ticker") if request.method=="GET"
           else request.form.get("ticker", ""))
    ticker = raw.strip().upper()

    # 2) Validate
    if not TICKER_RE.match(ticker):
        return jsonify({"error": "Invalid ticker format"}), 400

    # 3) Run your logic
    try:
        save_options_data(ticker)
        fig = calculate_and_visualize_data(ticker, width=1600, height=590)
        graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    except Exception as e:
        logging.exception("Error processing %s", ticker)
        return jsonify({"error": "Internal error, please try again"}), 500

    # 4) Safe cleanup
    BASE_DIR = os.getcwd()
    WORKDIR = os.path.join(BASE_DIR, "tmp_tickerdirs")
    os.makedirs(WORKDIR, exist_ok=True)

    # Paths we want to delete
    paths_to_remove = [
        os.path.join(BASE_DIR, ticker),  # legacy dir
        os.path.join(WORKDIR, ticker)  # new tmp dir
    ]

    for path in paths_to_remove:
        # Only remove if it lives under our app folder
        try:
            if os.path.commonpath([BASE_DIR, path]) == BASE_DIR:
                shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            logging.warning("Failed to clean up %s: %s", path, e)

    return jsonify({"ticker": ticker, "graph_json": graph_json})

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

