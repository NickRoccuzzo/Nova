import os
import json
import time
import random
import logging
import pandas as pd
import yfinance as yf
import threading

from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text

# ── Configuration ───────────────────────────────────────────────────────────────
DATABASE_URL      = "postgresql://option_user:option_pass@localhost:5432/tickers"
TICKERS_JSON_PATH = Path(__file__).parent / "tickers.json"
PROGRESS_FILE     = Path(__file__).parent / "progress.json"
STOP_EVENT        = threading.Event()
MIN_WAIT          = 1
MAX_WAIT          = 3

# ── Logging Setup ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class JSONProgressHandler(logging.Handler):
    """
    Handles INFO/ERROR logs from this file, updating progress.json.
    """
    def emit(self, record):
        # only handle logs originating in this file
        if not record.pathname.endswith(os.path.basename(__file__)):
            return

        msg = record.getMessage().strip()
        now = datetime.fromtimestamp(record.created)
        # cross-platform M/D HH:MM AM/PM
        if os.name == "nt":
            ts = now.strftime("%#m/%#d %I:%M %p")
        else:
            ts = now.strftime("%-m/%-d %I:%M %p")

        try:
            data = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            data = {}

        # ── TICKET EVENTS ─────────────────────────────────────────────────────────
        if msg.startswith("Processing ticker:"):
            ticker = msg.split("Processing ticker:")[1].strip()
            data.setdefault("tickers", {})[ticker] = {
                "status": "in-progress",
                "last_timestamp": ts
            }
        elif "Inserted" in msg and "rows for" in msg:
            parts  = msg.split("Inserted")[1].split("rows for")
            ticker = parts[1].strip()
            data.setdefault("tickers", {}).setdefault(ticker, {}).update({
                "status": "done",
                "last_timestamp": ts
            })

        # ── INDUSTRY EVENTS ───────────────────────────────────────────────────────
        elif msg.startswith("Starting industry:"):
            industry = msg.split("Starting industry:")[1].strip()
            data.setdefault("industries", {})[industry] = {
                "status": "in-progress",
                "last_timestamp": ts
            }
        elif msg.startswith("Finished industry:"):
            industry = msg.split("Finished industry:")[1].split(",")[0].strip()
            data.setdefault("industries", {})[industry] = {
                "status": "done",
                "last_timestamp": ts
            }

        # ── SECTOR EVENTS ─────────────────────────────────────────────────────────
        elif msg.startswith("Starting sector:"):
            sector = msg.split("Starting sector:")[1].strip()
            data.setdefault("sectors", {})[sector] = {
                "status": "in-progress",
                "last_timestamp": ts
            }
        elif msg.startswith("Finished sector:"):
            sector = msg.split("Finished sector:")[1].split(",")[0].strip()
            data.setdefault("sectors", {})[sector] = {
                "status": "done",
                "last_timestamp": ts
            }

        # ── ERRORS ─────────────────────────────────────────────────────────────────
        elif record.levelno >= logging.ERROR:
            data.setdefault("errors", []).append({
                "timestamp": ts,
                "message": msg
            })

        # ── SUMMARY ────────────────────────────────────────────────────────────────
        tickers = data.get("tickers", {})
        data["summary"] = {
            "completed": sum(1 for v in tickers.values() if v.get("status") == "done"),
            "total": len(tickers)
        }

        PROGRESS_FILE.write_text(json.dumps(data, indent=2))


# Attach handler
json_handler = JSONProgressHandler()
json_handler.setLevel(logging.INFO)
json_handler.setFormatter(logging.Formatter("%(message)s"))
root = logging.getLogger()
root.setLevel(logging.INFO)
root.addHandler(json_handler)

# Database engine
engine = create_engine(DATABASE_URL)


# ── Helper Functions ─────────────────────────────────────────────────────────────
def clear_progress():
    """Erase progress.json so a fresh run starts clean."""
    PROGRESS_FILE.write_text(json.dumps({}, indent=2))


def get_or_create_ticker_id(conn, symbol: str) -> int:
    res = conn.execute(text("SELECT ticker_id FROM tickers WHERE symbol = :sym"), {"sym": symbol}).fetchone()
    if res:
        return res[0]
    ins = conn.execute(
        text("INSERT INTO tickers (symbol) VALUES (:sym) RETURNING ticker_id"),
        {"sym": symbol}
    )
    return ins.fetchone()[0]


def get_or_create_expiration_id(conn, exp_date: pd.Timestamp) -> int:
    res = conn.execute(
        text("SELECT expiration_id FROM expirations WHERE expiration_date = :dt"),
        {"dt": exp_date.date()}
    ).fetchone()
    if res:
        return res[0]
    ins = conn.execute(
        text("INSERT INTO expirations (expiration_date) VALUES (:date) RETURNING expiration_id"),
        {"date": exp_date.date()}
    )
    return ins.fetchone()[0]


def fetch_and_clean_option_chain(symbol: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    all_frames = []
    keep_cols = [
        "expiration_date", "type", "strike",
        "last_trade_date", "last_price", "bid", "ask",
        "change_amt", "percent_change", "volume",
        "open_interest", "implied_volatility"
    ]

    for exp_str in ticker.options:
        try:
            opt = ticker.option_chain(exp_str)
            calls = opt.calls.copy()
            puts  = opt.puts.copy()
        except Exception as e:
            logger.error(f"Fetch error for {symbol} @ {exp_str}: {e}")
            continue

        calls["type"], puts["type"] = "CALL", "PUT"
        calls["expiration_date"] = pd.to_datetime(exp_str).date()
        puts["expiration_date"]  = pd.to_datetime(exp_str).date()

        for df in (calls, puts):
            df.rename(columns={
                "lastTradeDate":     "last_trade_date",
                "lastPrice":         "last_price",
                "percentChange":     "percent_change",
                "openInterest":      "open_interest",
                "impliedVolatility": "implied_volatility",
                "change":            "change_amt"
            }, inplace=True)

        calls = calls[keep_cols].copy()
        puts  = puts[keep_cols].copy()

        rounds = [
            ("last_price", 4), ("bid", 4), ("ask", 4),
            ("change_amt", 4), ("percent_change", 2),
            ("implied_volatility", 4),
        ]
        for col, prec in rounds:
            calls[col] = calls[col].round(prec)
            puts[col]  = puts[col].round(prec)

        all_frames.append(pd.concat([calls, puts], ignore_index=True))

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=keep_cols)


def upsert_on_conflict(table, conn, keys, data_iter):
    table_name    = table.name
    cols          = ", ".join(keys)
    vals          = ", ".join(f":{k}" for k in keys)
    conflict_keys = ["ticker_id", "expiration_id", "strike", "type"]
    update_cols   = [k for k in keys if k not in conflict_keys] + ["snapshot_timestamp"]

    set_clause = ", ".join(
        f"{col}=EXCLUDED.{col}" if col != "snapshot_timestamp" else "snapshot_timestamp=NOW()"
        for col in update_cols
    )

    sql = f"""
    INSERT INTO {table_name} ({cols})
    VALUES ({vals})
    ON CONFLICT (ticker_id, expiration_id, strike, type)
      DO UPDATE SET {set_clause}
    """
    for row in (dict(zip(keys, row)) for row in data_iter):
        conn.execute(text(sql), row)


# ── Helper (place this after upsert_on_conflict, before your main()) ──────────
def upsert_option_metrics(conn, ticker_id: int):
    """
    Compute per-expiry CALL/PUT sums + top-3 volumes, open_interest & strikes,
    then upsert into option_metrics for this single ticker.
    """
    conn.execute(text("""
    INSERT INTO option_metrics (
      ticker_id, expiration_id,
      call_vol_sum, put_vol_sum,
      max_vol_call,  max_call_oi,
      second_vol_call,  second_call_oi,
      third_vol_call,   third_call_oi,
      max_vol_put,   max_put_oi,
      second_vol_put,  second_put_oi,
      third_vol_put,   third_put_oi,
      max_call_strike,    second_call_strike,   third_call_strike,
      max_put_strike,     second_put_strike,    third_put_strike
    )
    SELECT
      pe.ticker_id,
      pe.expiration_id,
      pe.call_vol_sum,
      pe.put_vol_sum,

      -- top-3 CALL volumes & OI
      (SELECT oc.volume
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'call'
        ORDER BY oc.volume DESC NULLS LAST
        LIMIT 1
      ) AS max_vol_call,
      (SELECT oc.open_interest
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'call'
        ORDER BY oc.volume DESC NULLS LAST
        LIMIT 1
      ) AS max_call_oi,

      (SELECT oc.volume
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'call'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 1 LIMIT 1
      ) AS second_vol_call,
      (SELECT oc.open_interest
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'call'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 1 LIMIT 1
      ) AS second_call_oi,

      (SELECT oc.volume
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'call'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 2 LIMIT 1
      ) AS third_vol_call,
      (SELECT oc.open_interest
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'call'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 2 LIMIT 1
      ) AS third_call_oi,

      -- top-3 PUT volumes & OI
      (SELECT oc.volume
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'put'
        ORDER BY oc.volume DESC NULLS LAST
        LIMIT 1
      ) AS max_vol_put,
      (SELECT oc.open_interest
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'put'
        ORDER BY oc.volume DESC NULLS LAST
        LIMIT 1
      ) AS max_put_oi,

      (SELECT oc.volume
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'put'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 1 LIMIT 1
      ) AS second_vol_put,
      (SELECT oc.open_interest
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'put'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 1 LIMIT 1
      ) AS second_put_oi,

      (SELECT oc.volume
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'put'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 2 LIMIT 1
      ) AS third_vol_put,
      (SELECT oc.open_interest
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'put'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 2 LIMIT 1
      ) AS third_put_oi,

      -- matching top-3 CALL strikes
      (SELECT oc.strike
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'call'
        ORDER BY oc.volume DESC NULLS LAST
        LIMIT 1
      ) AS max_call_strike,
      (SELECT oc.strike
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'call'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 1 LIMIT 1
      ) AS second_call_strike,
      (SELECT oc.strike
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'call'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 2 LIMIT 1
      ) AS third_call_strike,

      -- matching top-3 PUT strikes
      (SELECT oc.strike
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'put'
        ORDER BY oc.volume DESC NULLS LAST
        LIMIT 1
      ) AS max_put_strike,
      (SELECT oc.strike
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'put'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 1 LIMIT 1
      ) AS second_put_strike,
      (SELECT oc.strike
         FROM option_contracts oc
        WHERE oc.ticker_id     = pe.ticker_id
          AND oc.expiration_id = pe.expiration_id
          AND LOWER(oc.type)   = 'put'
        ORDER BY oc.volume DESC NULLS LAST
        OFFSET 2 LIMIT 1
      ) AS third_put_strike

    FROM (
      SELECT
        ticker_id,
        expiration_id,
        SUM(volume) FILTER (WHERE LOWER(type)='call') AS call_vol_sum,
        SUM(volume) FILTER (WHERE LOWER(type)='put')  AS put_vol_sum
      FROM option_contracts
      WHERE ticker_id = :tid
      GROUP BY ticker_id, expiration_id
    ) AS pe

    ON CONFLICT (ticker_id, expiration_id)
      DO UPDATE SET
        call_vol_sum        = EXCLUDED.call_vol_sum,
        put_vol_sum         = EXCLUDED.put_vol_sum,
        max_vol_call        = EXCLUDED.max_vol_call,
        max_call_oi         = EXCLUDED.max_call_oi,
        second_vol_call     = EXCLUDED.second_vol_call,
        second_call_oi      = EXCLUDED.second_call_oi,
        third_vol_call      = EXCLUDED.third_vol_call,
        third_call_oi       = EXCLUDED.third_call_oi,
        max_vol_put         = EXCLUDED.max_vol_put,
        max_put_oi          = EXCLUDED.max_put_oi,
        second_vol_put      = EXCLUDED.second_vol_put,
        second_put_oi       = EXCLUDED.second_put_oi,
        third_vol_put       = EXCLUDED.third_vol_put,
        third_put_oi        = EXCLUDED.third_put_oi,
        max_call_strike     = EXCLUDED.max_call_strike,
        second_call_strike  = EXCLUDED.second_call_strike,
        third_call_strike   = EXCLUDED.third_call_strike,
        max_put_strike      = EXCLUDED.max_put_strike,
        second_put_strike   = EXCLUDED.second_put_strike,
        third_put_strike    = EXCLUDED.third_put_strike;
    """), {"tid": ticker_id})
# ── Main Logic ──────────────────────────────────────────────────────────────────
def main(full_run: bool = True):
    """
    full_run=True  → clear progress.json and run from start
    full_run=False → resume from existing progress.json
    """
    if full_run:
        clear_progress()
        now = datetime.now()
        run_ts = now.strftime("%#m/%#d %I:%M %p") if os.name == "nt" else now.strftime("%-m/%-d %I:%M %p")
        PROGRESS_FILE.write_text(json.dumps({
            "run_started": run_ts,
            "tickers": {},
            "industries": {},
            "sectors": {},
            "errors": []
        }, indent=2))

    # load what’s already done
    try:
        prev = json.loads(PROGRESS_FILE.read_text())
    except Exception:
        prev = {}
    done_tickers    = {tk for tk, v in prev.get("tickers",   {}).items() if v.get("status")   == "done"}
    done_industries = {i  for i,  v in prev.get("industries",{}).items() if v.get("status")   == "done"}
    done_sectors    = {s  for s,  v in prev.get("sectors",   {}).items() if v.get("status")   == "done"}

    nested = json.loads(TICKERS_JSON_PATH.read_text())
    failed_tickers = []

    for sector_name, industries in nested.items():
        if not full_run and sector_name in done_sectors:
            continue
        if STOP_EVENT.is_set():
            logging.info("Run stopped by user.")
            return
        logging.info(f"Starting sector: {sector_name}")

        for industry_name, entries in industries.items():
            if not full_run and industry_name in done_industries:
                continue
            if STOP_EVENT.is_set():
                logging.info("Run stopped by user.")
                return
            logging.info(f"  Starting industry: {industry_name}")

            for ent in entries:
                symbol    = ent["symbol"] if isinstance(ent, dict) else ent
                full_name = ent.get("full_name") if isinstance(ent, dict) else None

                if not full_run and symbol in done_tickers:
                    logging.info(f"    Skipping already-completed ticker: {symbol}")
                    continue
                if STOP_EVENT.is_set():
                    logging.info("Run stopped by user.")
                    return

                logging.info(f"    Processing ticker: {symbol}")
                try:
                    df = fetch_and_clean_option_chain(symbol)
                except Exception as fetch_err:
                    logging.error(f"    → Fetch error for {symbol}: {fetch_err}")
                    failed_tickers.append(symbol)
                    time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))
                    continue

                if df.empty:
                    logging.warning(f"    → No data for {symbol}, scheduling retry.")
                    failed_tickers.append(symbol)
                    time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))
                    continue

                try:
                    with engine.begin() as conn:
                        ticker_id = get_or_create_ticker_id(conn, symbol)
                        if full_name:
                            conn.execute(
                                text("""
                                    UPDATE tickers
                                       SET full_name = :fn
                                     WHERE ticker_id = :tid
                                       AND (full_name IS NULL OR full_name = '')
                                """),
                                {"fn": full_name, "tid": ticker_id}
                            )

                        df["expiration_date"] = pd.to_datetime(df["expiration_date"])
                        df["expiration_id"]   = df["expiration_date"].apply(
                            lambda dt: get_or_create_expiration_id(conn, pd.Timestamp(dt))
                        )
                        df["ticker_id"]       = ticker_id

                        df_to_insert = df[[
                            "ticker_id", "expiration_id", "strike", "type",
                            "last_trade_date", "last_price", "bid", "ask",
                            "change_amt", "percent_change", "volume",
                            "open_interest", "implied_volatility"
                        ]]

                        df_to_insert.to_sql(
                            "option_contracts",
                            con=conn,
                            if_exists="append",
                            index=False,
                            method=upsert_on_conflict
                        )

                        conn.execute(
                            text("UPDATE tickers SET last_queried_time = NOW() WHERE ticker_id = :tid"),
                            {"tid": ticker_id}
                        )
                        logging.info(f"      → Inserted {len(df_to_insert)} rows for {symbol}")

                        # now compute & upsert both volumes AND strikes
                        upsert_option_metrics(conn, ticker_id)
                        logging.info(f"      → option_metrics refreshed for {symbol}")

                except Exception as db_err:
                    logging.error(f"    → DB error for {symbol}: {db_err}")
                    failed_tickers.append(symbol)

                time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))

            logging.info(f"  Finished industry: {industry_name}, waiting before next industry…")
            time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))

        logging.info(f"Finished sector: {sector_name}, waiting before next sector…")
        time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))

    # ── Retry logic ──────────────────────────────────────────────────────────────
    if failed_tickers:
        logging.info(f"Retrying {len(failed_tickers)} failed tickers once more…")
        retry_list = failed_tickers.copy()
        failed_tickers.clear()

        for symbol in retry_list:
            if STOP_EVENT.is_set():
                logging.info("Run stopped by user.")
                return
            logging.info(f"  Retrying ticker: {symbol}")
            try:
                df = fetch_and_clean_option_chain(symbol)
                if df.empty:
                    logging.error(f"    → Still no data for {symbol} on retry.")
                    continue

                with engine.begin() as conn:
                    ticker_id = get_or_create_ticker_id(conn, symbol)
                    df["expiration_date"] = pd.to_datetime(df["expiration_date"])
                    df["expiration_id"]   = df["expiration_date"].apply(
                        lambda dt: get_or_create_expiration_id(conn, pd.Timestamp(dt))
                    )
                    df["ticker_id"]       = ticker_id

                    df_to_insert = df[[
                        "ticker_id", "expiration_id", "strike", "type",
                        "last_trade_date", "last_price", "bid", "ask",
                        "change_amt", "percent_change", "volume",
                        "open_interest", "implied_volatility"
                    ]]

                    df_to_insert.to_sql(
                        "option_contracts",
                        con=conn,
                        if_exists="append",
                        index=False,
                        method=upsert_on_conflict
                    )

                    conn.execute(
                        text("UPDATE tickers SET last_queried_time = NOW() WHERE ticker_id = :tid"),
                        {"tid": ticker_id}
                    )

                    logging.info(f"      → Retry inserted {len(df_to_insert)} rows for {symbol}")

                    # now also refresh option_metrics on retry
                    upsert_option_metrics(conn, ticker_id)
                    logging.info(f"      → (Retry) option_metrics refreshed for {symbol}")
                time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))
            except Exception as retry_err:
                logging.error(f"    → Final failure for {symbol}: {retry_err}")

    logging.info("All sectors/industries complete.  Exiting.")
