from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJ_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

DEFAULT_START = "2015-01-01"
DEFAULT_END   = "2025-06-30" #暫定 之後要新增deque

UNIVERSE = ["2330", "2317", "2454"]
