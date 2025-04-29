import os
import logging
import requests
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_URL = "https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{}.xlsx"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.ssga.com/us/en/intermediary/etfs/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
}
OUTPUT_DIR = "sectors"

def process_etf(etf: str) -> None:
    """
    Downloads and cleans holdings data for a given ETF.
    """
    url = BASE_URL.format(etf.lower())
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            logging.error(f"{etf}: Download failed (status code: {response.status_code}).")
            return

        df = pd.read_excel(response.content, skiprows=4)
        # Drop unnecessary columns
        df = df.drop(columns=["Identifier", "SEDOL", "Sector", "Local Currency", "Shares Held"], errors="ignore")
        # Keep only the top 10 holdings and format the 'Weight' column
        df = df.head(10)
        df["Weight"] = df["Weight"].round(2)

        cleaned_file_path = os.path.join(OUTPUT_DIR, f"{etf}_holdings.xlsx")
        df.to_excel(cleaned_file_path, index=False)
        logging.info(f"{etf}: Data cleaned and saved to {cleaned_file_path}.")
    except Exception as e:
        logging.error(f"{etf}: Error processing ETF holdings: {e}")


def validate_etf(etf: str) -> bool:
    """
    Check if the ETF holdings file exists at the source URL.
    """
    url = BASE_URL.format(etf.lower())
    try:
        # Allow redirects so that we can follow the 301 response.
        response = requests.head(url, headers=HEADERS, allow_redirects=True)
        if response.status_code == 200:
            return True
        else:
            logging.error(f"{etf}: Validation failed (status code: {response.status_code}).")
            return False
    except Exception as e:
        logging.error(f"{etf}: Error during ETF validation: {e}")
        return False


def enhance_holdings_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance ETF holdings data by:
      - Adding a "Daily Change (%)" column fetched via yfinance.
      - Converting the "Ticker" column into a clickable hyperlink that points to index.html with the ticker query parameter.

    Assumes the DataFrame has a "Ticker" column.
    """
    if "Ticker" in df.columns:
        changes = {}

        def get_daily_change(ticker: str):
            if ticker not in changes:
                try:
                    info = yf.Ticker(ticker).info
                    changes[ticker] = info.get("regularMarketChangePercent")
                except Exception as e:
                    changes[ticker] = None
            return changes[ticker]

        def format_change(change):
            if change is None:
                return "N/A"
            formatted = round(change, 2)
            if formatted > 0:
                return f'<span style="color: green;">+{formatted}%</span>'
            elif formatted < 0:
                return f'<span style="color: red;">{formatted}%</span>'
            else:
                return f'{formatted}%'

        # Compute daily change and format it.
        df["Daily Change (%)"] = df["Ticker"].apply(
            lambda t: format_change(get_daily_change(t))
        )

        # Change hyperlink to redirect to index.html with the ticker as a query parameter
        df["Ticker"] = df["Ticker"].apply(
            lambda t: f'<a href="/?ticker={t}" target="_self">{t}</a>'
        )
    else:
        df["Daily Change (%)"] = "N/A"
    return df

# Example usage when updating the holdings:
def update_holdings(etfs=None) -> None:
    if etfs is None:
        etfs = ["XLRE", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLC", "XLV", "XLF", "XBI"]

    # Filter ETFs that are valid according to the validation function
    valid_etfs = [etf for etf in etfs if validate_etf(etf)]

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(process_etf, valid_etfs)

    logging.info("All valid ETF holdings have been processed.")

if __name__ == "__main__":
    update_holdings()
