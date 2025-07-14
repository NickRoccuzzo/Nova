# File: query_then_upsert.py

from query_options import options_dictionary, unusual_volume_report
from upsert_options import upsert_rows, upsert_unusual_report

upsert_rows(options_dictionary)
upsert_unusual_report(unusual_volume_report)

print("Done.")

