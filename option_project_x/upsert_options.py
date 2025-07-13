# File: store_option_chains.py

import os
from sqlalchemy import (
    create_engine, MetaData, Table,
    Column, String, Float, Integer
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

# 1) Engine & metadata
DB_URL = os.getenv("DB_URL", "sqlite:///./options.db")
engine = create_engine(DB_URL, echo=False)
metadata = MetaData()

# 2) Main option_chain table (with sector & industry)
option_chain = Table(
    # Basic building blocks
    "option_chain", metadata,
    Column("ticker",             String,  primary_key=True, nullable=False),
    Column("expiration_date",    String,  primary_key=True, nullable=False),
    Column("sector",             String),
    Column("industry",           String),
    # Open-Interest (OI)-sorted
    Column("call_strike_OI",     Float),    # Call Strike
    Column("call_volume_OI",     Integer),  # Call Volume
    Column("call_OI_OI",         Integer),  # Call OI
    Column("put_strike_OI",      Float),    # Put Strike
    Column("put_volume_OI",      Integer),  # Put Volume
    Column("put_OI_OI",          Integer),  # Put OI
    Column("call_OI_sum",        Integer),  # Call OI sum
    Column("put_OI_sum",         Integer),  # Put OI sum
    # Volume-sorted
    Column("call_strike_vol",    Float),
    Column("call_volume_vol",    Integer),
    Column("call_OI_vol",        Integer),
    Column("put_strike_vol",     Float),
    Column("put_volume_vol",     Integer),
    Column("put_OI_vol",         Integer),
    Column("call_vol_sum",       Float),
    Column("put_vol_sum",        Float),
    # Unusual-Report strings to use
    Column("call_unusualness",   String),
    Column("put_unusualness",    String),
)

# 3) Unusual volume report table
unusual_vol = Table(
    "unusual_volume_report", metadata,
    Column("ticker",             String,  primary_key=True, nullable=False),
    Column("expiration_date",    String,  primary_key=True, nullable=False),
    Column("side",               String,  primary_key=True, nullable=False),
    Column("sector",             String),
    Column("industry",           String),
    Column("strike",             Float),
    Column("volume",             Integer),
    Column("openInterest",       Integer),
    Column("unusualness",        String),
)

# 4) Create tables if missing
metadata.create_all(engine)

# 5) Upsert main table
def upsert_rows(rows):
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
                    index_elements=["ticker","expiration_date"],
                    set_={c: getattr(stmt.excluded, c)
                          for c in data if c not in ("ticker","expiration_date")}
                )

            conn.execute(stmt)

# 6) Upsert unusual report
def upsert_unusual_report(report_rows):
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
                    index_elements=["ticker","expiration_date","side"],
                    set_={c: getattr(stmt.excluded, c)
                          for c in data if c not in ("ticker","expiration_date","side")}
                )

            conn.execute(stmt)
