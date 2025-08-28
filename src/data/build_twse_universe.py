"""
Build a fixed TWSE universe (common shares only) with completeness < 1%.
Outputs:
  - universe_YYYY-MM-DD_to_YYYY-MM-DD.csv  (full list)
  - universe_table.md                      (top N rows, paste to Medium)
Dependencies: requests, pandas, tqdm
Data source: FinMind API (v4)
"""

import os, time, math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from tqdm import tqdm

# ---------- CONFIG (edit here) ----------
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
MISSING_PCT_MAX = 0.01     # completeness < 1%
TOP_N_FOR_MD = 50          # how many rows to preview in Medium
MAX_WORKERS = 8            # API fan-out; tune to avoid rate limits 
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")  # set env var if you have a token
# ---------------------------------------
