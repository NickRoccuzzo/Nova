import sqlite3
import pandas as pd

# 1) Connect to the SQLite file
conn = sqlite3.connect("options.db")

# 2) Quick sanity check: what tables do you have?
tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)
print("Tables in DB:\n", tables)

# 3) Pull out the first few rows of your option_chain table
df = pd.read_sql_query(
    "SELECT * FROM option_chain ORDER BY expiration_date LIMIT 10;",
    conn
)
print(df)