import random

import pandas as pd
import pytest

from phantomlink import arena, classifier, data


def _tiny_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Small synthetic dataset with a learnable lexical difference between classes."""
    rng = random.Random(seed)
    words = ["river", "garden", "studio", "market", "north", "coffee", "travel", "school",
             "music", "green", "photo", "design", "maple", "ocean", "craft", "books"]
    bait = ["secure", "login", "verify", "account", "update", "wallet"]
    rows = []
    for i in range(n):
        phish = i % 2
        if phish:
            name = f"{rng.choice(bait)}-{rng.choice(words)}{rng.randint(10, 99999)}.web"
        else:
            name = rng.choice(words) + rng.choice(words)
        reg = f"{name.split('.')[-1]}{i}.com"
        rows.append({"name": name, "registered": reg, "suffix": "com", "is_phishing": phish,
                     "split": ("train", "train", "train", "val", "test")[i % 5]})
    return pd.DataFrame(rows).drop_duplicates("name").reset_index(drop=True)


@pytest.fixture
def tiny_dataset(monkeypatch):
    df = _tiny_df()
    monkeypatch.setattr(data, "load_processed", lambda *a, **k: df)
    classifier._split_xy.cache_clear()
    classifier.class_medians.cache_clear()
    arena.dataset_index.cache_clear()
    yield df
    classifier._split_xy.cache_clear()
    classifier.class_medians.cache_clear()
    arena.dataset_index.cache_clear()
