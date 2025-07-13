# import pull_option_chains; pull_option_chains.run()

from query_option_chains import options_dictionary, unusual_volume_report
from upsert_option_chains_in_db import upsert_rows, upsert_unusual_report

upsert_rows(options_dictionary)
upsert_unusual_report(unusual_volume_report)

print("Done.")