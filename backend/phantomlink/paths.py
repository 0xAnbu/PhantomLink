"""Filesystem layout. Every path can be overridden with PHANTOMLINK_HOME."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("PHANTOMLINK_HOME", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
PROCESSED_CSV = DATA_DIR / "processed" / "domains.csv"
SPLIT_REPORT = DATA_DIR / "processed" / "split_report.json"

CHECKPOINT_DIR = ROOT / "checkpoints"          # shipped example checkpoints
RUNS_DIR = ROOT / "runs"                        # checkpoints produced from the UI
FRONTEND_DIST = ROOT / "frontend" / "dist"

DATASET_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
