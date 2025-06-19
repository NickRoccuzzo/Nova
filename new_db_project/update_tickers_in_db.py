import json
import logging

from sqlalchemy import create_engine, text

# ——————————————————————————————————————————————————————————————————————————————
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ——————————————————————————————————————————————————————————————————————————————

def main():
    # 1) Load our single master JSON
    with open("tickers.json") as f:
        sector_data = json.load(f)

    # 2) Extract every symbol + any full_name from JSON
    json_symbols = set()
    json_name_map = {}
    for industries in sector_data.values():
        for entries in industries.values():
            for e in entries:
                if isinstance(e, dict):
                    sym = e["symbol"]
                    json_symbols.add(sym)
                    if e.get("full_name"):
                        json_name_map[sym] = e["full_name"]
                else:
                    json_symbols.add(e)

    # 3) Connect
    engine = create_engine(
        "postgresql://option_user:option_pass@localhost:5432/tickers",
        echo=False, future=True
    )

    with engine.begin() as conn:
        # — Pre-sync: fetch existing tickers —
        res = conn.execute(text("SELECT symbol, full_name FROM tickers"))
        db_rows = res.all()
        db_symbols   = {row[0] for row in db_rows}
        db_name_map  = {row[0]: row[1] for row in db_rows}

        #  Determine what to insert / delete / update
        to_insert = json_symbols - db_symbols
        to_delete = db_symbols - json_symbols
        to_update = [
            sym for sym in (json_symbols & db_symbols)
            if sym in json_name_map and json_name_map[sym] != db_name_map.get(sym)
        ]

        #  DELETE removed symbols
        if to_delete:
            conn.execute(
                text("DELETE FROM tickers WHERE symbol = ANY(:symbols)"),
                {"symbols": list(to_delete)}
            )
            logger.info(f"Deleted {len(to_delete)} tickers: {to_delete}")

        #  INSERT new symbols (with any known full_name)
        if to_insert:
            insert_rows = [
                {"symbol": sym, "full_name": json_name_map.get(sym)}
                for sym in to_insert
            ]
            conn.execute(
                text("INSERT INTO tickers(symbol, full_name) VALUES (:symbol, :full_name)"),
                insert_rows
            )
            logger.info(f"Inserted {len(to_insert)} new tickers.")

        #  UPDATE changed full_names
        for sym in to_update:
            conn.execute(
                text("UPDATE tickers SET full_name = :full_name WHERE symbol = :symbol"),
                {"symbol": sym, "full_name": json_name_map[sym]}
            )
        if to_update:
            logger.info(f"Updated full_name for {len(to_update)} tickers.")

        # — create the lookup tables & columns if they don’t exist —
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sectors (
              sector_id   SERIAL PRIMARY KEY,
              sector_name TEXT UNIQUE NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS industries (
              industry_id   SERIAL PRIMARY KEY,
              industry_name TEXT NOT NULL,
              sector_id     INT NOT NULL REFERENCES sectors(sector_id),
              UNIQUE(industry_name, sector_id)
            );
        """))
        conn.execute(text("ALTER TABLE tickers ADD COLUMN IF NOT EXISTS industry_id INT;"))
        conn.execute(text("ALTER TABLE tickers ADD COLUMN IF NOT EXISTS full_name   TEXT;"))
        conn.execute(text("""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1
              FROM pg_constraint
             WHERE conname = 'fk_ticker_industry'
               AND conrelid = 'tickers'::regclass
          ) THEN
            ALTER TABLE tickers
              ADD CONSTRAINT fk_ticker_industry
                FOREIGN KEY(industry_id)
                REFERENCES industries(industry_id)
              DEFERRABLE INITIALLY IMMEDIATE;
          END IF;
        END
        $$;
        """))

        # 4) Walk the JSON to upsert sectors → industries → assign industry_id
        for sector_name, industries in sector_data.items():
            # upsert sector
            conn.execute(text("""
                INSERT INTO sectors(sector_name)
                VALUES (:sector_name)
                ON CONFLICT(sector_name) DO NOTHING;
            """), {"sector_name": sector_name})
            sector_id = conn.execute(text("""
                SELECT sector_id FROM sectors
                 WHERE sector_name = :sector_name
            """), {"sector_name": sector_name}).scalar_one()

            for industry_name, entries in industries.items():
                # upsert industry
                conn.execute(text("""
                    INSERT INTO industries(industry_name, sector_id)
                    VALUES (:industry_name, :sector_id)
                    ON CONFLICT(industry_name, sector_id) DO NOTHING;
                """), {"industry_name": industry_name, "sector_id": sector_id})
                industry_id = conn.execute(text("""
                    SELECT industry_id FROM industries
                     WHERE industry_name = :industry_name
                       AND sector_id     = :sector_id
                """), {"industry_name": industry_name, "sector_id": sector_id}).scalar_one()

                # collect just the symbols for this industry
                symbols = [
                    e["symbol"] if isinstance(e, dict) else e
                    for e in entries
                ]
                # bulk‐assign industry_id
                conn.execute(text("""
                    UPDATE tickers
                       SET industry_id = :industry_id
                     WHERE symbol = ANY(:symbols)
                """), {"industry_id": industry_id, "symbols": symbols})

    logger.info("✅ Full sync complete! Database now mirrors tickers.json.")

if __name__ == "__main__":
    main()
