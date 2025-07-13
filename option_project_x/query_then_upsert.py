# import pull_option_chains; pull_option_chains.run()

from query_option_chains import options_dictionary
from upsert_option_chains_in_db import upsert_rows

upsert_rows(options_dictionary)

print("Done.")