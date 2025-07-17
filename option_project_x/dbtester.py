import sqlite3, pandas as pd

conn = sqlite3.connect("options.db")
df   = pd.read_sql_query(
    "SELECT ticker, expiration_date, side, total_spent "
    "FROM unusual_volume_report "
    "WHERE ticker='MOS';",
    conn
)
conn.close()
print(df)