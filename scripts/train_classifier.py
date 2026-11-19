"""Train the RandomForest detector from the command line and save it as a checkpoint.

    python scripts/train_classifier.py                 # -> checkpoints/classifier_default.joblib
    python scripts/train_classifier.py --n-estimators 200 --out runs/classifier_big.joblib
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from phantomlink import classifier, paths

if __name__ == "__main__":
    d = classifier.ClassifierConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-estimators", type=int, default=d.n_estimators)
    ap.add_argument("--max-depth", type=int, default=d.max_depth)
    ap.add_argument("--min-samples-leaf", type=int, default=d.min_samples_leaf)
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--threshold", type=float, default=d.threshold)
    ap.add_argument("--train-size", type=int, default=d.train_size)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", type=Path, default=paths.CHECKPOINT_DIR / "classifier_default.joblib")
    a = ap.parse_args()
    cfg = classifier.ClassifierConfig(
        n_estimators=a.n_estimators, max_depth=a.max_depth, min_samples_leaf=a.min_samples_leaf,
        class_weight=None if a.class_weight == "none" else a.class_weight,
        threshold=a.threshold, train_size=a.train_size, seed=a.seed)

    def log(e):
        if e["type"] == "stage":
            print("·", e["message"])
        elif e["type"] == "metrics":
            print(f"  trees={e['trees']:4d}  val_f1={e['val_f1']:.4f}  val_auc={e['val_auc']:.4f}")

    bundle = classifier.train(cfg, on_event=log)
    bundle.save(a.out)
    m = bundle.metrics
    print(json.dumps({"val": m["val"], "test": m["test"]}, indent=2))
    print(f"Saved {a.out}")
