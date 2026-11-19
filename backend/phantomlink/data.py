"""Dataset preparation for PHANTOMLINK.

Source: PhiUSIIL Phishing URL Dataset (UCI ML Repository, id 967, CC BY 4.0).

IMPORTANT label semantics of the raw CSV:  label == 1 -> LEGITIMATE, label == 0 -> PHISHING.
Internally we always use ``is_phishing`` (1 = phishing) to avoid ambiguity.

Only the domain string is used. Page-content attributes (title, forms, iframes, ...) are
discarded because the demo never visits a site and would not have them at inference time.

Normalisation (each step removes a shortcut that is not a lexical property of the name):
  * lowercase, strip port / trailing dot
  * drop IP-address hosts (no name to analyse)
  * strip a leading ``www.`` — in this dataset 100 % of legitimate hosts start with www.
    versus ~41 % of phishing hosts, so keeping it would leak the label
  * drop the public suffix (TLD). The generator emits names under the reserved ``.test``
    TLD, so the detector must judge the same thing in training and in the Arena: the name.

Splits are assigned by *registered domain* (eTLD+1, ICANN section of the Public Suffix
List bundled with tldextract — no network access) using a stable hash, so every host of
the same registered domain lands in the same split.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import paths

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-."
_VALID_NAME = re.compile(r"^[a-z0-9.-]+$")
_LABEL = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
FICTITIOUS_TLD = "test"  # RFC 2606 reserved TLD: can never resolve on the public Internet


@lru_cache(maxsize=1)
def _extractor():
    import tldextract

    # suffix_list_urls=() -> use the snapshot bundled with the package, never the network.
    return tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


@dataclass(frozen=True)
class NormalizedDomain:
    name: str          # what both models see, e.g. "secure-login.paypa1"
    registered: str    # eTLD+1 used for group splitting, e.g. "paypa1.com"
    suffix: str        # original public suffix (kept for the report only)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def normalize_host(raw: str) -> NormalizedDomain | None:
    """Normalise a raw hostname. Returns None when it cannot be used."""
    if not isinstance(raw, str):
        return None
    host = raw.strip().lower().rstrip(".")
    host = host.split("/")[0].split("@")[-1]
    if host.count(":") == 1:  # host:port (IPv6 has several colons and is rejected below)
        host = host.split(":")[0]
    if not host or _is_ip(host) or ":" in host:
        return None
    ext = _extractor()(host)
    if not ext.suffix or not ext.domain:
        return None
    sub = ext.subdomain
    while sub == "www" or sub.startswith("www."):  # also "www.www.example.com"
        sub = sub[4:]
    name = f"{sub}.{ext.domain}" if sub else ext.domain
    if not _VALID_NAME.match(name):
        return None
    return NormalizedDomain(name=name, registered=f"{ext.domain}.{ext.suffix}", suffix=ext.suffix)


def is_valid_name(name: str) -> bool:
    """Syntactic validity of a generated name (labels per RFC 1035 / 1123, LDH rule)."""
    if not name or len(name) > 253 - len(FICTITIOUS_TLD) - 1:
        return False
    return all(_LABEL.match(label) for label in name.split("."))


def split_of(registered: str) -> str:
    """Deterministic split assignment from a hash of the registered domain."""
    bucket = int(hashlib.sha256(registered.encode()).hexdigest()[:8], 16) % 10_000 / 10_000
    if bucket < SPLIT_FRACTIONS["train"]:
        return "train"
    if bucket < SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]:
        return "val"
    return "test"


def prepare(raw_csv: Path = paths.RAW_CSV, out_csv: Path = paths.PROCESSED_CSV,
            report_path: Path = paths.SPLIT_REPORT) -> dict:
    """Build data/processed/domains.csv and a JSON report describing every filtering step."""
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run `python scripts/prepare_data.py --download` first.")
    raw = pd.read_csv(raw_csv, usecols=["Domain", "label"])
    report: dict = {"source": "PhiUSIIL Phishing URL Dataset (UCI id 967)",
                    "raw_rows": int(len(raw)),
                    "raw_label_counts": {"legitimate(label=1)": int((raw.label == 1).sum()),
                                         "phishing(label=0)": int((raw.label == 0).sum())}}

    raw["is_phishing"] = (raw["label"] == 0).astype(int)  # <- the one place labels are mapped
    raw["had_www"] = raw["Domain"].str.lower().str.startswith("www.")
    report["www_prefix_rate"] = {
        "legitimate": round(float(raw.loc[raw.is_phishing == 0, "had_www"].mean()), 4),
        "phishing": round(float(raw.loc[raw.is_phishing == 1, "had_www"].mean()), 4),
    }

    norm = raw["Domain"].map(normalize_host)
    report["dropped_unparseable_or_ip"] = int(norm.isna().sum())
    raw = raw[norm.notna()].copy()
    norm = norm[norm.notna()]
    raw["name"] = [n.name for n in norm]
    raw["registered"] = [n.registered for n in norm]
    raw["suffix"] = [n.suffix for n in norm]

    # Same normalised name with both labels -> ambiguous, drop all of them.
    label_variety = raw.groupby("name")["is_phishing"].nunique()
    conflicting = set(label_variety[label_variety > 1].index)
    report["dropped_conflicting_names"] = len(conflicting)
    raw = raw[~raw["name"].isin(conflicting)]
    before = len(raw)
    raw = raw.drop_duplicates("name")
    report["dropped_duplicates"] = int(before - len(raw))

    raw["split"] = raw["registered"].map(split_of)
    out = raw[["name", "registered", "suffix", "is_phishing", "split"]].sort_values(
        ["split", "name"]).reset_index(drop=True)

    # Sanity: no registered domain may appear in two splits.
    leaks = out.groupby("registered")["split"].nunique()
    assert int((leaks > 1).sum()) == 0, "registered domain present in several splits"

    report["final_rows"] = int(len(out))
    report["splits"] = {
        s: {"rows": int((g := out[out.split == s]).shape[0]),
            "phishing": int(g.is_phishing.sum()),
            "legitimate": int((g.is_phishing == 0).sum()),
            "registered_domains": int(g.registered.nunique())}
        for s in ("train", "val", "test")
    }
    biggest = out.groupby("registered").size().sort_values(ascending=False).head(10)
    report["largest_registered_groups"] = {k: int(v) for k, v in biggest.items()}
    report["registered_domain_overlap_between_splits"] = 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    report_path.write_text(json.dumps(report, indent=2))
    return report


@lru_cache(maxsize=2)
def _load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, keep_default_na=False, dtype={"name": str})


def load_processed(csv_path: Path = paths.PROCESSED_CSV) -> pd.DataFrame:
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"{csv_path} not found. Run `python scripts/prepare_data.py`.")
    return _load(str(csv_path))
