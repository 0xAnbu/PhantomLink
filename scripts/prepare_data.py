"""Download (optional) and prepare the PhiUSIIL dataset.

    python scripts/prepare_data.py --download
"""
import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from phantomlink import data, paths  # noqa: E402


def download() -> None:
    paths.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {paths.DATASET_URL} ...")
    with urllib.request.urlopen(paths.DATASET_URL, timeout=120) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        member = next(n for n in z.namelist() if n.endswith(".csv"))
        paths.RAW_CSV.write_bytes(z.read(member))
    print(f"Saved {paths.RAW_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download the raw CSV from UCI")
    args = ap.parse_args()
    if args.download or not paths.RAW_CSV.exists():
        download()
    print(json.dumps(data.prepare(), indent=2))
