import json
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from historical import get_latest_emas
from seeker import get_tickers_by_sector  # assumes seeker.py is in your PYTHONPATH

# ─── Logging ───────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─── Config ────────────────────────────────
TICKERS_JSON = Path(__file__).parent / "tickers.json"
MAX_WORKERS  = 4     # adjust up / down based on your rate‑limit tolerance
PAUSE_SEC    = 1     # small pause between batches

def flatten_all_tickers(path: Path) -> list[str]:
    with open(path, "r") as f:
        data = json.load(f)
    all_tickers = []
    for sector, industries in data.items():
        for industry, tickers in industries.items():
            all_tickers.extend(tickers)
    return sorted(set(all_tickers))

def update_ticker(ticker: str):
    try:
        emas = get_latest_emas(ticker)
        logging.info(f"[✓] {ticker}: EMAs updated → {emas}")
    except Exception as e:
        logging.error(f"[✗] {ticker}: failed to update EMAs: {e}")

def main():
    # 1) grab the full list
    tickers = flatten_all_tickers(TICKERS_JSON)
    logging.info(f"Starting EMA update for {len(tickers)} tickers…")

    # 2) concurrent fetch
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(update_ticker, t): t for t in tickers}
        for fut in as_completed(futures):
            # Throttle a bit so you don’t slam Yahoo all at once
            time.sleep(PAUSE_SEC)

    logging.info("EMA update run complete.")

if __name__ == "__main__":
    main()