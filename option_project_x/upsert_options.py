# store_option_chains.py
import os
from sqlalchemy import (
    create_engine, MetaData, Table,
    Column, String, Float, Integer
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

#    SQLite (local file):  sqlite:///./options.db
DB_URL = os.getenv("DB_URL", "sqlite:///./options.db")
engine = create_engine(DB_URL, echo=False)
metadata = MetaData()

# // Define your table schema
option_chain = Table(
    "option_chain", metadata,
    Column("ticker",         String, primary_key=True),
    Column("expiration_date",       String,  primary_key=True),
    Column("call_strike_OI",        Float),
    Column("call_volume_OI",        Integer),
    Column("call_OI_OI",            Integer),
    Column("put_strike_OI",         Float),
    Column("put_volume_OI",         Integer),
    Column("put_OI_OI",             Integer),
    Column("call_OI_sum",           Integer),
    Column("put_OI_sum",            Integer),
    Column("call_strike_vol",       Float),
    Column("call_volume_vol",       Integer),
    Column("call_OI_vol",           Integer),
    Column("put_strike_vol",        Float),
    Column("put_volume_vol",        Integer),
    Column("put_OI_vol",            Integer),
    Column("call_vol_sum",          Float),
    Column("put_vol_sum",           Float),
    Column("call_unusualness",      String),
    Column("put_unusualness",       String),
)

unusual_vol = Table(
  "unusual_volume_report", metadata,
Column("ticker",          String,  primary_key=True, nullable=False),
  Column("expiration_date", String, primary_key=True),
  Column("side",            String, primary_key=True),
  Column("strike",          Float),
  Column("volume",          Integer),
  Column("openInterest",    Integer),
  Column("unusualness",     String),
)

# // Create table if missing
metadata.create_all(engine)

def upsert_rows(rows):
    """Upsert a list of dicts into option_chain on expiration_date."""
    with engine.begin() as conn:
        for r in rows:
            # Map your dictionary to the table columns
            data = {
                "ticker": r["ticker"],
                "expiration_date": r["expiration_date"],
                # unpack the two tuples
                "call_strike_OI":   r["call_contract_with_largest_OI"][0],
                "call_volume_OI":   r["call_contract_with_largest_OI"][1],
                "call_OI_OI":       r["call_contract_with_largest_OI"][2],
                "put_strike_OI":    r["put_contract_with_largest_OI"][0],
                "put_volume_OI":    r["put_contract_with_largest_OI"][1],
                "put_OI_OI":        r["put_contract_with_largest_OI"][2],
                "call_OI_sum":      r["call_options_OI_sum"],
                "put_OI_sum":       r["put_options_OI_sum"],
                "call_strike_vol":  r["call_with_the_largest_volume"][0],
                "call_volume_vol":  r["call_with_the_largest_volume"][1],
                "call_OI_vol":      r["call_with_the_largest_volume"][2],
                "put_strike_vol":   r["put_with_the_largest_volume"][0],
                "put_volume_vol":   r["put_with_the_largest_volume"][1],
                "put_OI_vol":       r["put_with_the_largest_volume"][2],
                "call_vol_sum":     r["call_options_volume_sum"],
                "put_vol_sum":      r["put_options_volume_sum"],
                "call_unusualness": r["call_unusualness"],
                "put_unusualness":  r["put_unusualness"],
            }

            if DB_URL.startswith("sqlite"):
                # SQLite: INSERT OR REPLACE
                stmt = option_chain.insert().prefix_with("OR REPLACE").values(**data)
            else:
                # Postgres: ON CONFLICT DO UPDATE
                stmt = pg_insert(option_chain).values(**data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["ticker", "expiration_date"],
                    set_={col: getattr(stmt.excluded, col)
                          for col in data if col not in ("ticker", "expiration_date")}
                )
            conn.execute(stmt)


def upsert_unusual_report(report_rows):
    # no thresholds here—just write whatever rows you get
    with engine.begin() as conn:
        for r in report_rows:
            data = {
                "ticker":          r["ticker"],
                "expiration_date": r["expiration_date"],
                "side":            r["side"],
                "strike":          r["strike"],
                "volume":          r["volume"],
                "openInterest":    r["openInterest"],
                "unusualness":     r["unusualness"],
            }
            # use your INSERT…OR REPLACE or ON CONFLICT logic
            stmt = unusual_vol.insert().prefix_with("OR REPLACE").values(**data)
            conn.execute(stmt)


if __name__ == "__main__":
    # assume your `options_dictionary` is built/imported here
    from query_options import options_dictionary
    upsert_rows(options_dictionary)
    print("Upserted", len(options_dictionary), "rows into", DB_URL)