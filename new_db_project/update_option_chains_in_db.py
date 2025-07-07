import os
import json
import time
import random
import logging
import pandas as pd
import yfinance as yf
import threading
from db_config import POSTGRES_DB_URL
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text
from datetime import datetime, timezone

# ── Configuration ───────────────────────────────────────────────────────────────
TICKERS_JSON_PATH = Path(__file__).parent / "tickers.json"
STOP_EVENT = threading.Event()
MIN_WAIT = 1
MAX_WAIT = 3

# ── Logging Setup ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Database engine
engine = create_engine(POSTGRES_DB_URL)

# ── Helper Functions ─────────────────────────────────────────────────────────────

def prune_expired_data(conn):
    """
    Delete all option and metric rows whose expiration_date is before today,
    then delete those expirations themselves.
    """
    # 1. Delete raw option_contracts for old expirations
    conn.execute(text("""
      DELETE FROM option_contracts oc
      USING expirations e
      WHERE oc.expiration_id = e.expiration_id
        AND e.expiration_date < CURRENT_DATE;
    """))

    # 2. Delete option_metrics for old expirations
    conn.execute(text("""
      DELETE FROM option_metrics om
      USING expirations e
      WHERE om.expiration_id = e.expiration_id
        AND e.expiration_date < CURRENT_DATE;
    """))

    # 3. Delete the expired rows from expirations
    conn.execute(text("""
      DELETE FROM expirations
      WHERE expiration_date < CURRENT_DATE;
    """))

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

def upsert_option_metrics(conn, ticker_id: int):
    conn.execute(text("""
    WITH
    per_expiry AS (
      SELECT
        oc.ticker_id,
        oc.expiration_id,
        SUM(volume) FILTER (WHERE LOWER(type) = 'call') AS call_vol_sum,
        SUM(volume) FILTER (WHERE LOWER(type) = 'put')  AS put_vol_sum
      FROM option_contracts oc
      WHERE oc.ticker_id = :tid
      GROUP BY oc.ticker_id, oc.expiration_id
    ),

    -- rank calls by open_interest DESC
    calls_ranked AS (
      SELECT
        oc.ticker_id,
        oc.expiration_id,
        oc.strike,
        oc.last_price,
        oc.volume,
        oc.open_interest AS oi,
        ROW_NUMBER() OVER (
          PARTITION BY oc.ticker_id, oc.expiration_id
          ORDER BY oc.open_interest DESC
        ) AS rn
      FROM option_contracts oc
      WHERE oc.ticker_id = :tid
        AND LOWER(oc.type) = 'call'
    ),

    -- same for puts
    puts_ranked AS (
      SELECT
        oc.ticker_id,
        oc.expiration_id,
        oc.strike,
        oc.last_price,
        oc.volume,
        oc.open_interest AS oi,
        ROW_NUMBER() OVER (
          PARTITION BY oc.ticker_id, oc.expiration_id
          ORDER BY oc.open_interest DESC
        ) AS rn
      FROM option_contracts oc
      WHERE oc.ticker_id = :tid
        AND LOWER(oc.type) = 'put'
    ),

    call_ratio AS (
      SELECT
        oc.ticker_id,
        oc.expiration_id,
        AVG(oc.volume::NUMERIC / NULLIF(oc.open_interest,0)) AS avg_ratio
      FROM option_contracts oc
      WHERE oc.ticker_id = :tid
        AND LOWER(oc.type) = 'call'
      GROUP BY oc.ticker_id, oc.expiration_id
    ),
    put_ratio AS (
      SELECT
        oc.ticker_id,
        oc.expiration_id,
        AVG(oc.volume::NUMERIC / NULLIF(oc.open_interest,0)) AS avg_ratio
      FROM option_contracts oc
      WHERE oc.ticker_id = :tid
        AND LOWER(oc.type) = 'put'
      GROUP BY oc.ticker_id, oc.expiration_id
    )

    INSERT INTO option_metrics (
      ticker_id, expiration_id,
      call_vol_sum, put_vol_sum,

      -- top-1 call
      max_call_strike,   max_call_last_price,   max_call_volume,   max_call_oi,
      unusual_max_vol_call,      unusual_max_vol_call_score,
      -- top-2 call
      second_call_strike, second_call_last_price, second_call_volume, second_call_oi,
      unusual_second_vol_call,   unusual_second_vol_call_score,
      -- top-3 call
      third_call_strike,  third_call_last_price,  third_call_volume,  third_call_oi,
      unusual_third_vol_call,    unusual_third_vol_call_score,

      -- top-1 put
      max_put_strike,    max_put_last_price,    max_put_volume,    max_put_oi,
      unusual_max_vol_put,       unusual_max_vol_put_score,
      -- top-2 put
      second_put_strike, second_put_last_price, second_put_volume, second_put_oi,
      unusual_second_vol_put,    unusual_second_vol_put_score,
      -- top-3 put
      third_put_strike,  third_put_last_price,  third_put_volume,  third_put_oi,
      unusual_third_vol_put,     unusual_third_vol_put_score

    )
    SELECT
      pe.ticker_id,
      pe.expiration_id,
      pe.call_vol_sum,
      pe.put_vol_sum,

      /* === CALL side === */
      -- 1st
      MAX(CASE WHEN cr.rn=1 THEN cr.strike END)         AS max_call_strike,
      MAX(CASE WHEN cr.rn=1 THEN cr.last_price END)     AS max_call_last_price,
      MAX(CASE WHEN cr.rn=1 THEN cr.volume END)         AS max_call_volume,
      MAX(CASE WHEN cr.rn=1 THEN cr.oi END)             AS max_call_oi,
      BOOL_OR(cr.rn=1 AND cr.volume>cr.oi)              AS unusual_max_vol_call,
      LEAST(10, GREATEST(0,
        ROUND(10.0 * (
          (MAX(CASE WHEN cr.rn=1 THEN cr.volume END)::NUMERIC
           / NULLIF(MAX(CASE WHEN cr.rn=1 THEN cr.oi END),0))
          / COALESCE(cr1.avg_ratio,1)
        ))
      )) AS unusual_max_vol_call_score,

      -- 2nd
      MAX(CASE WHEN cr.rn=2 THEN cr.strike END)         AS second_call_strike,
      MAX(CASE WHEN cr.rn=2 THEN cr.last_price END)     AS second_call_last_price,
      MAX(CASE WHEN cr.rn=2 THEN cr.volume END)         AS second_call_volume,
      MAX(CASE WHEN cr.rn=2 THEN cr.oi END)             AS second_call_oi,
      BOOL_OR(cr.rn=2 AND cr.volume>cr.oi)              AS unusual_second_vol_call,
      LEAST(10, GREATEST(0,
        ROUND(10.0 * (
          (MAX(CASE WHEN cr.rn=2 THEN cr.volume END)::NUMERIC
           / NULLIF(MAX(CASE WHEN cr.rn=2 THEN cr.oi END),0))
          / COALESCE(cr1.avg_ratio,1)
        ))
      )) AS unusual_second_vol_call_score,

      -- 3rd
      MAX(CASE WHEN cr.rn=3 THEN cr.strike END)         AS third_call_strike,
      MAX(CASE WHEN cr.rn=3 THEN cr.last_price END)     AS third_call_last_price,
      MAX(CASE WHEN cr.rn=3 THEN cr.volume END)         AS third_call_volume,
      MAX(CASE WHEN cr.rn=3 THEN cr.oi END)             AS third_call_oi,
      BOOL_OR(cr.rn=3 AND cr.volume>cr.oi)              AS unusual_third_vol_call,
      LEAST(10, GREATEST(0,
        ROUND(10.0 * (
          (MAX(CASE WHEN cr.rn=3 THEN cr.volume END)::NUMERIC
           / NULLIF(MAX(CASE WHEN cr.rn=3 THEN cr.oi END),0))
          / COALESCE(cr1.avg_ratio,1)
        ))
      )) AS unusual_third_vol_call_score,

      /* === PUT side === */
      -- 1st
      MAX(CASE WHEN pr.rn=1 THEN pr.strike END)         AS max_put_strike,
      MAX(CASE WHEN pr.rn=1 THEN pr.last_price END)     AS max_put_last_price,
      MAX(CASE WHEN pr.rn=1 THEN pr.volume END)         AS max_put_volume,
      MAX(CASE WHEN pr.rn=1 THEN pr.oi END)             AS max_put_oi,
      BOOL_OR(pr.rn=1 AND pr.volume>pr.oi)              AS unusual_max_vol_put,
      LEAST(10, GREATEST(0,
        ROUND(10.0 * (
          (MAX(CASE WHEN pr.rn=1 THEN pr.volume END)::NUMERIC
           / NULLIF(MAX(CASE WHEN pr.rn=1 THEN pr.oi END),0))
          / COALESCE(pr1.avg_ratio,1)
        ))
      )) AS unusual_max_vol_put_score,

      -- 2nd
      MAX(CASE WHEN pr.rn=2 THEN pr.strike END)         AS second_put_strike,
      MAX(CASE WHEN pr.rn=2 THEN pr.last_price END)     AS second_put_last_price,
      MAX(CASE WHEN pr.rn=2 THEN pr.volume END)         AS second_put_volume,
      MAX(CASE WHEN pr.rn=2 THEN pr.oi END)             AS second_put_oi,
      BOOL_OR(pr.rn=2 AND pr.volume>pr.oi)              AS unusual_second_vol_put,
      LEAST(10, GREATEST(0,
        ROUND(10.0 * (
          (MAX(CASE WHEN pr.rn=2 THEN pr.volume END)::NUMERIC
           / NULLIF(MAX(CASE WHEN pr.rn=2 THEN pr.oi END),0))
          / COALESCE(pr1.avg_ratio,1)
        ))
      )) AS unusual_second_vol_put_score,

      -- 3rd
      MAX(CASE WHEN pr.rn=3 THEN pr.strike END)         AS third_put_strike,
      MAX(CASE WHEN pr.rn=3 THEN pr.last_price END)     AS third_put_last_price,
      MAX(CASE WHEN pr.rn=3 THEN pr.volume END)         AS third_put_volume,
      MAX(CASE WHEN pr.rn=3 THEN pr.oi END)             AS third_put_oi,
      BOOL_OR(pr.rn=3 AND pr.volume>pr.oi)              AS unusual_third_vol_put,
      LEAST(10, GREATEST(0,
        ROUND(10.0 * (
          (MAX(CASE WHEN pr.rn=3 THEN pr.volume END)::NUMERIC
           / NULLIF(MAX(CASE WHEN pr.rn=3 THEN pr.oi END),0))
          / COALESCE(pr1.avg_ratio,1)
        ))
      )) AS unusual_third_vol_put_score

    FROM per_expiry pe
    LEFT JOIN calls_ranked cr  ON cr.ticker_id=pe.ticker_id  AND cr.expiration_id=pe.expiration_id
    LEFT JOIN call_ratio  cr1 ON cr1.ticker_id=pe.ticker_id AND cr1.expiration_id=pe.expiration_id
    LEFT JOIN puts_ranked pr  ON pr.ticker_id=pe.ticker_id  AND pr.expiration_id=pe.expiration_id
    LEFT JOIN put_ratio   pr1 ON pr1.ticker_id=pe.ticker_id AND pr1.expiration_id=pe.expiration_id

    GROUP  BY
      pe.ticker_id,
      pe.expiration_id,
      pe.call_vol_sum,
      pe.put_vol_sum,
      cr1.avg_ratio,
      pr1.avg_ratio

    ON CONFLICT (ticker_id, expiration_id) DO UPDATE SET
      call_vol_sum                   = EXCLUDED.call_vol_sum,
      put_vol_sum                    = EXCLUDED.put_vol_sum,

      max_call_strike                = EXCLUDED.max_call_strike,
      max_call_last_price            = EXCLUDED.max_call_last_price,
      max_call_volume                = EXCLUDED.max_call_volume,
      max_call_oi                    = EXCLUDED.max_call_oi,
      unusual_max_vol_call           = EXCLUDED.unusual_max_vol_call,
      unusual_max_vol_call_score     = EXCLUDED.unusual_max_vol_call_score,

      second_call_strike             = EXCLUDED.second_call_strike,
      second_call_last_price         = EXCLUDED.second_call_last_price,
      second_call_volume             = EXCLUDED.second_call_volume,
      second_call_oi                 = EXCLUDED.second_call_oi,
      unusual_second_vol_call        = EXCLUDED.unusual_second_vol_call,
      unusual_second_vol_call_score  = EXCLUDED.unusual_second_vol_call_score,

      third_call_strike              = EXCLUDED.third_call_strike,
      third_call_last_price          = EXCLUDED.third_call_last_price,
      third_call_volume              = EXCLUDED.third_call_volume,
      third_call_oi                  = EXCLUDED.third_call_oi,
      unusual_third_vol_call         = EXCLUDED.unusual_third_vol_call,
      unusual_third_vol_call_score   = EXCLUDED.unusual_third_vol_call_score,

      max_put_strike                 = EXCLUDED.max_put_strike,
      max_put_last_price             = EXCLUDED.max_put_last_price,
      max_put_volume                 = EXCLUDED.max_put_volume,
      max_put_oi                     = EXCLUDED.max_put_oi,
      unusual_max_vol_put            = EXCLUDED.unusual_max_vol_put,
      unusual_max_vol_put_score      = EXCLUDED.unusual_max_vol_put_score,

      second_put_strike              = EXCLUDED.second_put_strike,
      second_put_last_price          = EXCLUDED.second_put_last_price,
      second_put_volume              = EXCLUDED.second_put_volume,
      second_put_oi                  = EXCLUDED.second_put_oi,
      unusual_second_vol_put         = EXCLUDED.unusual_second_vol_put,
      unusual_second_vol_put_score   = EXCLUDED.unusual_second_vol_put_score,

      third_put_strike               = EXCLUDED.third_put_strike,
      third_put_last_price           = EXCLUDED.third_put_last_price,
      third_put_volume               = EXCLUDED.third_put_volume,
      third_put_oi                   = EXCLUDED.third_put_oi,
      unusual_third_vol_put          = EXCLUDED.unusual_third_vol_put,
      unusual_third_vol_put_score    = EXCLUDED.unusual_third_vol_put_score;
    """), {"tid": ticker_id})

def upsert_unusual_events(conn, ticker_id: int):
    """
    Push all currently-unusual metrics into unusual_option_events
    then delete any rows there that are no longer marked unusual.
    """
    # 1) upsert all rows that are flagged unusual in option_metrics
    conn.execute(text("""
        INSERT INTO unusual_option_events (
          ticker_id, expiration_id,
          unusual_max_vol_call,         unusual_max_vol_call_score,
          unusual_second_vol_call,      unusual_second_vol_call_score,
          unusual_third_vol_call,       unusual_third_vol_call_score,
          unusual_max_vol_put,          unusual_max_vol_put_score,
          unusual_second_vol_put,       unusual_second_vol_put_score,
          unusual_third_vol_put,        unusual_third_vol_put_score
        )
        SELECT
          m.ticker_id,
          m.expiration_id,
          COALESCE(m.unusual_max_vol_call, FALSE),
          COALESCE(m.unusual_max_vol_call_score, 0),
          COALESCE(m.unusual_second_vol_call, FALSE),
          COALESCE(m.unusual_second_vol_call_score, 0),
          COALESCE(m.unusual_third_vol_call, FALSE),
          COALESCE(m.unusual_third_vol_call_score, 0),
          COALESCE(m.unusual_max_vol_put, FALSE),
          COALESCE(m.unusual_max_vol_put_score, 0),
          COALESCE(m.unusual_second_vol_put, FALSE),
          COALESCE(m.unusual_second_vol_put_score, 0),
          COALESCE(m.unusual_third_vol_put, FALSE),
          COALESCE(m.unusual_third_vol_put_score, 0)
        FROM option_metrics m
        WHERE m.ticker_id = :tid
          AND (
               m.unusual_max_vol_call
            OR m.unusual_second_vol_call
            OR m.unusual_third_vol_call
            OR m.unusual_max_vol_put
            OR m.unusual_second_vol_put
            OR m.unusual_third_vol_put
          )
        ON CONFLICT (ticker_id, expiration_id) DO UPDATE SET
          unusual_max_vol_call         = EXCLUDED.unusual_max_vol_call,
          unusual_max_vol_call_score   = EXCLUDED.unusual_max_vol_call_score,
          unusual_second_vol_call      = EXCLUDED.unusual_second_vol_call,
          unusual_second_vol_call_score= EXCLUDED.unusual_second_vol_call_score,
          unusual_third_vol_call       = EXCLUDED.unusual_third_vol_call,
          unusual_third_vol_call_score = EXCLUDED.unusual_third_vol_call_score,
          unusual_max_vol_put          = EXCLUDED.unusual_max_vol_put,
          unusual_max_vol_put_score    = EXCLUDED.unusual_max_vol_put_score,
          unusual_second_vol_put       = EXCLUDED.unusual_second_vol_put,
          unusual_second_vol_put_score = EXCLUDED.unusual_second_vol_put_score,
          unusual_third_vol_put        = EXCLUDED.unusual_third_vol_put,
          unusual_third_vol_put_score  = EXCLUDED.unusual_third_vol_put_score;
    """), {"tid": ticker_id})

    # 2) delete any events that are no longer marked unusual
    conn.execute(text("""
    DELETE FROM unusual_option_events u
    WHERE u.ticker_id = :tid
      AND NOT EXISTS (
        SELECT 1
        FROM option_metrics m
        WHERE m.ticker_id     = u.ticker_id
          AND m.expiration_id = u.expiration_id
          AND (
               m.unusual_max_vol_call
            OR m.unusual_second_vol_call
            OR m.unusual_third_vol_call
            OR m.unusual_max_vol_put
            OR m.unusual_second_vol_put
            OR m.unusual_third_vol_put
          )
      );
    """), {"tid": ticker_id})

def upsert_ticker_metrics(conn, ticker_id: int):
    # sum up option_metrics for this ticker
    conn.execute(text("""
    INSERT INTO ticker_metrics
      (ticker_id,
       call_oi_total, put_oi_total,
       call_vol_total, put_vol_total,
       call_iv_total,  put_iv_total)
    SELECT
      ticker_id,
      SUM(call_oi_sum),
      SUM(put_oi_sum),
      SUM(call_vol_sum),
      SUM(put_vol_sum),
      SUM(call_iv_sum),
      SUM(put_iv_sum)
    FROM option_metrics
    WHERE ticker_id = :tid
    GROUP BY ticker_id
    ON CONFLICT (ticker_id)
      DO UPDATE SET
        call_oi_total   = EXCLUDED.call_oi_total,
        put_oi_total    = EXCLUDED.put_oi_total,
        call_vol_total  = EXCLUDED.call_vol_total,
        put_vol_total   = EXCLUDED.put_vol_total,
        call_iv_total   = EXCLUDED.call_iv_total,
        put_iv_total    = EXCLUDED.put_iv_total
    """), {"tid": ticker_id})

def upsert_industry_metrics(conn, ticker_id: int):
    # lookup industry_id for this ticker, then sum ticker_metrics
    conn.execute(text("""
    WITH t AS (
      SELECT industry_id FROM tickers WHERE ticker_id = :tid
    )
    INSERT INTO industry_metrics
      (industry_id,
       industry_call_oi, industry_put_oi,
       industry_call_vol, industry_put_vol,
       industry_call_iv,  industry_put_iv)
    SELECT
      t.industry_id,
      SUM(tm.call_oi_total),
      SUM(tm.put_oi_total),
      SUM(tm.call_vol_total),
      SUM(tm.put_vol_total),
      SUM(tm.call_iv_total),
      SUM(tm.put_iv_total)
    FROM ticker_metrics tm
    JOIN tickers USING (ticker_id)
    JOIN t USING (industry_id)
    GROUP BY t.industry_id
    ON CONFLICT (industry_id)
      DO UPDATE SET
        industry_call_oi  = EXCLUDED.industry_call_oi,
        industry_put_oi   = EXCLUDED.industry_put_oi,
        industry_call_vol = EXCLUDED.industry_call_vol,
        industry_put_vol  = EXCLUDED.industry_put_vol,
        industry_call_iv  = EXCLUDED.industry_call_iv,
        industry_put_iv   = EXCLUDED.industry_put_iv
    """), {"tid": ticker_id})

def upsert_sector_metrics(conn, ticker_id: int):
    # lookup sector_id for this ticker’s industry, then sum industry_metrics
    conn.execute(text("""
    WITH sec AS (
      SELECT i.sector_id
      FROM industries i
      JOIN tickers t ON t.industry_id = i.industry_id
      WHERE t.ticker_id = :tid
    )
    INSERT INTO sector_metrics
      (sector_id,
       sector_call_oi, sector_put_oi,
       sector_call_vol, sector_put_vol,
       sector_call_iv,  sector_put_iv)
    SELECT
      sec.sector_id,
      SUM(im.industry_call_oi),
      SUM(im.industry_put_oi),
      SUM(im.industry_call_vol),
      SUM(im.industry_put_vol),
      SUM(im.industry_call_iv),
      SUM(im.industry_put_iv)
    FROM industry_metrics im
    JOIN industries i USING (industry_id)
    JOIN sec       USING (sector_id)
    GROUP BY sec.sector_id
    ON CONFLICT (sector_id)
      DO UPDATE SET
        sector_call_oi  = EXCLUDED.sector_call_oi,
        sector_put_oi   = EXCLUDED.sector_put_oi,
        sector_call_vol = EXCLUDED.sector_call_vol,
        sector_put_vol  = EXCLUDED.sector_put_vol,
        sector_call_iv  = EXCLUDED.sector_call_iv,
        sector_put_iv   = EXCLUDED.sector_put_iv
    """), {"tid": ticker_id})

def flatten_symbols() -> list:
    nested = json.loads(TICKERS_JSON_PATH.read_text())
    symbols = []
    for industries in nested.values():
        for entries in industries.values():
            for e in entries:
                symbols.append(e["symbol"] if isinstance(e, dict) else e)
    return symbols

def load_last_queried_times() -> dict:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT symbol, last_queried_time FROM tickers")).fetchall()
    return {r[0]: r[1] for r in rows}

def refresh_matview(conn):
    conn.execute(text("REFRESH MATERIALIZED VIEW unusual_events_ranked;"))


def refresh_ticker_zscores(conn):
    conn.execute(text("REFRESH MATERIALIZED VIEW ticker_metrics_zscores"))
def refresh_industry_zscores(conn):
    conn.execute(text("REFRESH MATERIALIZED VIEW industry_metrics_zscores"))
def refresh_sector_zscores(conn):
    conn.execute(text("REFRESH MATERIALIZED VIEW sector_metrics_zscores"))

def refresh_unusual_volume_report(conn):
    conn.execute(text("REFRESH MATERIALIZED VIEW unusual_volume_report"))

def refresh_industry_unusual_report(conn):
    conn.execute(text("REFRESH MATERIALIZED VIEW industry_unusual_report"))

def refresh_sector_unusual_report(conn):
    conn.execute(text("REFRESH MATERIALIZED VIEW sector_unusual_report"))

# ── Main Logic ──────────────────────────────────────────────────────────────────


def main():
    """
    Runs one cycle: prune expired data, then either a fresh full run
    (if last cycle completed) or resume based on last_queried_time.
    """
    # 1) Prune any truly expired expirations & their data
    with engine.begin() as conn:
        prune_expired_data(conn)

    # 2) Count how many tickers are still unqueried
    with engine.connect() as conn:
        remaining = conn.execute(
            text("SELECT COUNT(*) FROM tickers WHERE last_queried_time IS NULL")
        ).scalar()

    # 3) If none remain, clear all timestamps for a fresh sweep
    if remaining == 0:
        logger.info("All tickers have timestamps → clearing last_queried_time for a fresh cycle")
        with engine.begin() as conn:
            conn.execute(text("UPDATE tickers SET last_queried_time = NULL"))
    else:
        logger.info(f"Resuming cycle: {remaining} tickers left to fetch")

    # 4) Load the timestamps & flatten symbol list
    last_times  = load_last_queried_times()
    all_symbols = flatten_symbols()

    # 5) Sort so unqueried come first (None→0), then oldest timestamp
    all_symbols.sort(
        key=lambda s: last_times.get(s).timestamp() if last_times.get(s) else 0
    )

    failed = []

    for symbol in all_symbols:
        # skip any symbol already done this cycle
        if last_times.get(symbol) is not None:
            continue

        if STOP_EVENT.is_set():
            logger.info("Run stopped by user.")
            break

        logger.info(f"Processing ticker: {symbol}")

        # get ticker_id up front so we can stamp it even on no-data
        with engine.begin() as conn:
            tid = get_or_create_ticker_id(conn, symbol)

        # --- FETCH ---
        try:
            df = fetch_and_clean_option_chain(symbol)
        except Exception as e:
            logger.error(f"Fetch error for {symbol}: {e}")
            # mark as queried so we don't retry forever
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE tickers SET last_queried_time = NOW() WHERE ticker_id = :tid"),
                    {"tid": tid},
                )
            last_times[symbol] = datetime.now(timezone.utc)
            failed.append(symbol)
            time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))
            continue

        # --- NO DATA ---
        if df.empty:
            logger.warning(f"No data for {symbol}, marking as queried.")
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE tickers SET last_queried_time = NOW() WHERE ticker_id = :tid"),
                    {"tid": tid},
                )
            last_times[symbol] = datetime.now(timezone.utc)
            # do NOT append to failed; it’s intentionally skipped
            time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))
            continue

        # --- SUCCESSFUL INGEST & METRICS REFRESH ---
        try:
            with engine.begin() as conn:
                # map expirations & insert raw contracts
                df["expiration_date"] = pd.to_datetime(df["expiration_date"])
                df["expiration_id"]   = df["expiration_date"].apply(
                    lambda dt: get_or_create_expiration_id(conn, pd.Timestamp(dt))
                )
                df["ticker_id"]       = tid

                df_to_ins = df[
                    [
                        "ticker_id", "expiration_id", "strike", "type",
                        "last_trade_date", "last_price", "bid", "ask",
                        "change_amt", "percent_change", "volume",
                        "open_interest", "implied_volatility"
                    ]
                ]
                df_to_ins.to_sql(
                    "option_contracts",
                    con=conn,
                    if_exists="append",
                    index=False,
                    method=upsert_on_conflict
                )

                # stamp the ticker as done
                conn.execute(
                    text("UPDATE tickers SET last_queried_time = NOW() WHERE ticker_id = :tid"),
                    {"tid": tid},
                )
                last_times[symbol] = datetime.now(timezone.utc)

                # refresh all downstream metrics & events
                upsert_option_metrics(conn, tid)
                logger.info(f"option_metrics refreshed for {symbol}")

                upsert_unusual_events(conn, tid)
                logger.info(f"unusual events upserted for {symbol}")

                upsert_ticker_metrics(conn, tid)
                upsert_industry_metrics(conn, tid)
                upsert_sector_metrics(conn, tid)

                logger.info(f"Inserted/updated {len(df_to_ins)} rows for {symbol}")

                refresh_matview(conn)
                logger.info("Unusual Events refreshed successfully")
                refresh_sector_zscores(conn)
                logger.info("Sector Z-scores refreshed successfully")
                refresh_industry_zscores(conn)
                logger.info("Industry Z-scores refreshed successfully")
                refresh_ticker_zscores(conn)
                logger.info("Ticker Z-scores refreshed successfully")

                refresh_unusual_volume_report(conn)
                logger.info("Unusual Volume Report Updated!")
                refresh_sector_unusual_report(conn)
                logger.info("Unusual Sectors Updated!")
                refresh_industry_unusual_report(conn)
                logger.info("Unusual Industries Updated!")

        except Exception as e:
            logger.error(f"DB error for {symbol}: {e}")
            failed.append(symbol)

        # pause between tickers
        time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))

    # 6) Report on any failures in this cycle
    if failed:
        logger.warning(f"{len(failed)} tickers failed this cycle: {failed}")

    logger.info("All tickers processed. Exiting.")

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception:
            logger.exception("Unexpected crash in main(); retrying in 5 minutes")

        logger.info("Sleeping 5 minutes before next cycle…")
        time.sleep(5 * 60)


