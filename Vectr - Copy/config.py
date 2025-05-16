# config.py

import os
import zoneinfo

MAX_TOP_CONTRACTS_OVERALL  = 4
MAX_TOP_CONTRACTS_PER_EXPIRY = 2
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "…")
local_tz = zoneinfo.ZoneInfo("America/New_York")