# File: upsert_options.py

import os
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Integer
from sqlalchemy.dialects.postgresql import insert as pg_insert

DB_URL = os.getenv("DB_URL", "sqlite:///./options.db")
engine = create_engine(DB_URL, echo=False)
metadata = MetaData()

# ─── option_chain ───────────────────────────────────────────────────────────────
option_chain = Table(
    "option_chain", metadata,
    Column("ticker",         String,  primary_key=True, nullable=False),
    Column("expiration_date",String,  primary_key=True, nullable=False),
    Column("sector",         String),
    Column("industry",       String),
    Column("call_strike_OI", Float),
    Column("call_volume_OI", Integer),
    Column("call_OI_OI",     Integer),
    Column("put_strike_OI",  Float),
    Column("put_volume_OI",  Integer),
    Column("put_OI_OI",      Integer),
    Column("call_OI_sum",    Integer),
    Column("put_OI_sum",     Integer),
    Column("call_strike_vol",Float),
    Column("call_volume_vol",Integer),
    Column("call_OI_vol",    Integer),
    Column("put_strike_vol", Float),
    Column("put_volume_vol", Integer),
    Column("put_OI_vol",     Integer),
    Column("call_vol_sum",   Float),
    Column("put_vol_sum",    Float),
    Column("call_unusualness",String),
    Column("put_unusualness", String),
)

# ─── unusual_volume_report ─────────────────────────────────────────────────────
unusual_vol = Table(
    "unusual_volume_report", metadata,
    Column("ticker",         String,  primary_key=True, nullable=False),
    Column("expiration_date",String,  primary_key=True, nullable=False),
    Column("side",           String,  primary_key=True, nullable=False),
    Column("sector",         String),
    Column("industry",       String),
    Column("strike",         Float),
    Column("volume",         Integer),
    Column("openInterest",   Integer),
    Column("unusualness",    String),
)

# ─── market_structure_report ───────────────────────────────────────────────────
market_structure = Table(
    "market_structure_report", metadata,
    Column("ticker",           String,  primary_key=True, nullable=False),
    Column("expiration_date",  String,  primary_key=True, nullable=False),
    Column("sector",           String),
    Column("industry",         String),
    Column("structure",        String),
    Column("pct_diff",         Float),
    Column("avg_oi",           Float),
    Column("days_to_expiry",   Integer),
    Column("pairedness",       Float),
    Column("final_score",      Float),
)

# Create any missing tables
metadata.create_all(engine)


def upsert_rows(rows):
    """Upsert option_chain rows."""
    with engine.begin() as conn:
        for r in rows:
            data = {
                "ticker":            r["ticker"],
                "expiration_date":   r["expiration_date"],
                "sector":            r["sector"],
                "industry":          r["industry"],
                "call_strike_OI":    r["call_contract_with_largest_OI"][0],
                "call_volume_OI":    r["call_contract_with_largest_OI"][1],
                "call_OI_OI":        r["call_contract_with_largest_OI"][2],
                "put_strike_OI":     r["put_contract_with_largest_OI"][0],
                "put_volume_OI":     r["put_contract_with_largest_OI"][1],
                "put_OI_OI":         r["put_contract_with_largest_OI"][2],
                "call_OI_sum":       r["call_options_OI_sum"],
                "put_OI_sum":        r["put_options_OI_sum"],
                "call_strike_vol":   r["call_with_the_largest_volume"][0],
                "call_volume_vol":   r["call_with_the_largest_volume"][1],
                "call_OI_vol":       r["call_with_the_largest_volume"][2],
                "put_strike_vol":    r["put_with_the_largest_volume"][0],
                "put_volume_vol":    r["put_with_the_largest_volume"][1],
                "put_OI_vol":        r["put_with_the_largest_volume"][2],
                "call_vol_sum":      r["call_options_volume_sum"],
                "put_vol_sum":       r["put_options_volume_sum"],
                "call_unusualness":  r["call_unusualness"],
                "put_unusualness":   r["put_unusualness"],
            }
            if DB_URL.startswith("sqlite"):
                stmt = option_chain.insert().prefix_with("OR REPLACE").values(**data)
            else:
                stmt = pg_insert(option_chain).values(**data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["ticker", "expiration_date"],
                    set_={c: getattr(stmt.excluded, c)
                          for c in data if c not in ("ticker", "expiration_date")}
                )
            conn.execute(stmt)


def upsert_unusual_report(report_rows):
    """Upsert unusual_volume_report rows."""
    with engine.begin() as conn:
        for r in report_rows:
            data = {
                "ticker":          r["ticker"],
                "expiration_date": r["expiration_date"],
                "side":            r["side"],
                "sector":          r["sector"],
                "industry":        r["industry"],
                "strike":          r["strike"],
                "volume":          r["volume"],
                "openInterest":    r["openInterest"],
                "unusualness":     r["unusualness"],
            }
            if DB_URL.startswith("sqlite"):
                stmt = unusual_vol.insert().prefix_with("OR REPLACE").values(**data)
            else:
                stmt = pg_insert(unusual_vol).values(**data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["ticker", "expiration_date", "side"],
                    set_={c: getattr(stmt.excluded, c)
                          for c in data if c not in ("ticker", "expiration_date", "side")}
                )
            conn.execute(stmt)


def upsert_market_structure(rows):
    """Upsert market_structure_report rows."""
    with engine.begin() as conn:
        for r in rows:
            data = {
                "ticker":           r["ticker"],
                "expiration_date":  r["expiration_date"],
                "sector":           r.get("sector"),
                "industry":         r.get("industry"),
                "structure":        r["structure"],
                "pct_diff":         r["pct_diff"],
                "avg_oi":           r["avg_oi"],
                "days_to_expiry":   r["days_to_expiry"],
                "pairedness":       r["pairedness"],
                "final_score":      r["final_score"],
            }
            if DB_URL.startswith("sqlite"):
                stmt = market_structure.insert().prefix_with("OR REPLACE").values(**data)
            else:
                stmt = pg_insert(market_structure).values(**data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["ticker", "expiration_date"],
                    set_={c: getattr(stmt.excluded, c)
                          for c in data if c not in ("ticker", "expiration_date")}
                )
            conn.execute(stmt)