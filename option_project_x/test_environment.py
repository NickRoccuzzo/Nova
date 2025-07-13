import pandas as pd
import sqlite3

conn = sqlite3.connect("options.db")
# view your tables
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn))

# load the main table
df = pd.read_sql("SELECT * FROM option_chain LIMIT 10;", conn)
print(df)
