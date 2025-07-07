import pandas as pd
from sqlalchemy import create_engine
from db_config import POSTGRES_DB_URL

engine = create_engine(POSTGRES_DB_URL)
df = pd.read_sql("SELECT * FROM unusual_volume_report", engine)
print(df.shape)
print(df.head())