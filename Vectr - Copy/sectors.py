import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_etf_performance(etfs):
    performance = {}
    today = datetime.today().date()

    # Define time intervals
    timeframes = {
        "1-day": today - timedelta(days=1),
        "1-week": today - timedelta(weeks=1),
        "1-month": today - timedelta(days=30),
        "3-month": today - timedelta(days=90),
        "year-to-date": datetime(today.year, 1, 1).date(),
        "1-year": today - timedelta(days=365),
        "5-year": today - timedelta(days=1825),
        "max": None,  # Special case for max performance
    }

    for etf in etfs:
        try:
            # Fetch historical data
            data = yf.Ticker(etf).history(period="max")
            if data.empty:
                raise ValueError(f"No price data found for {etf}. Symbol may be delisted or unavailable.")

            data.index = pd.to_datetime(data.index).date  # Ensure the index is in date format
            latest_close = data["Close"].iloc[-1]

            etf_performance = {}
            for label, date in timeframes.items():
                if date is not None:
                    # Find the nearest date before or on the target date
                    nearest_date = max([d for d in data.index if d <= date], default=None)
                    if nearest_date is not None:
                        previous_close = data.loc[nearest_date, "Close"]
                    else:
                        previous_close = None
                else:  # Max performance
                    previous_close = data["Close"].iloc[0]

                # Calculate performance if valid data is found
                if previous_close:
                    performance_change = ((latest_close - previous_close) / previous_close) * 100
                    etf_performance[label] = round(performance_change, 2)
                else:
                    etf_performance[label] = "N/A"

            performance[etf] = etf_performance
        except Exception as e:
            # Log the error and display a user-friendly message
            print(f"{etf}: {e}")
            performance[etf] = {"error": "Data not available"}

    return performance

if __name__ == "__main__":
    etfs = ["XLRE", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLC", "XLV", "XLF", "XBI"]
    print(get_etf_performance(etfs))
