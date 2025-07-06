# db_config.py

# ! This is a CRITICAL file that is referenced by all files in the project for database and other configurations !

# PostgreSQL Database Configuration:
DB_CONFIG = {
    "HOST": "localhost",
    "PORT": 5432,
    "DB_TYPE": "postgresql",
    "DB_NAME": "tickers",
    "USER": "option_user",
    "PASSWORD": "option_pass"
}

# EMA PERIODS (used primarily in 'update_emas_in_db.py' and 'update_ema_scenarios_in_db.py')
EMA_CONFIG = {
    "PERIODS": [3, 5, 7, 9, 12, 15, 18, 21, 25, 29, 33, 37, 42, 47, 50, 52, 57, 75, 85, 95, 100, 105, 115, 125, 150, 200]
}

# Master file where all tickers, along with their associated industry/sectors are located and configured
TICKERS_FILE = "tickers.json"

# ___ FULL URL ___
POSTGRES_DB_URL = f"{DB_CONFIG['DB_TYPE']}://{DB_CONFIG['USER']}:{DB_CONFIG['PASSWORD']}@{DB_CONFIG['HOST']}:{DB_CONFIG['PORT']}/{DB_CONFIG['DB_NAME']}"

