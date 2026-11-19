"""Model B — RandomForest detector over lexical features.

Training grows the forest incrementally (warm_start) so the UI can watch validation metrics
evolve as trees are added. Final metrics are reported on the held-out *test* split, which is
never used for training or for any decision during training.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import data, features

FEATURE_VERSION = 1


@dataclass
class ClassifierConfig:
    n_estimators: int = 100
    max_depth: int | None = 12
    min_samples_leaf: int = 2
    class_weight: str | None = "balanced"   # None | "balanced"
    threshold: float = 0.5
    train_size: int = 30_000                 # rows sampled from the train split
    seed: int = 42
    n_jobs: int = 2


@dataclass
class ClassifierBundle:
    model: RandomForestClassifier
    config: ClassifierConfig
    feature_names: list[str]
    metrics: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    feature_version: int = FEATURE_VERSION
    label_semantics: str = "positive class (1) = phishing-like; raw PhiUSIIL label 0 = phishing"
    sklearn_version: str = sklearn.__version__

    @property
    def threshold(self) -> float:
        return self.config.threshold

    def predict_proba(self, names: list[str]) -> np.ndarray:
        """Probability of the phishing-like class for each name."""
        if not names:
            return np.zeros(0)
        return self.model.predict_proba(features.matrix(names))[:, 1]

    def tree_votes(self, name: str) -> list[float]:
        """Per-tree phishing probability for one name (spread = forest disagreement)."""
        x = features.matrix([name])
        return [float(t.predict_proba(x)[0, 1]) for t in self.model.estimators_]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> "ClassifierBundle":
        bundle = joblib.load(path)
        if not isinstance(bundle, ClassifierBundle):
            raise TypeError(f"{path} is not a PHANTOMLINK classifier bundle")
        if bundle.feature_version != FEATURE_VERSION:
            raise ValueError("checkpoint built with an incompatible feature extractor")
        return bundle


@lru_cache(maxsize=4)
def _split_xy(split: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    df = data.load_processed()
    part = df[df.split == split]
    names = tuple(part["name"].tolist())
    return features.matrix(names), part["is_phishing"].to_numpy(dtype=np.int64), names


def metrics_at(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
        "n": int(len(y_true)),
    }


def _histogram(y: np.ndarray, proba: np.ndarray, bins: int = 20) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    legit, _ = np.histogram(proba[y == 0], bins=edges)
    phish, _ = np.histogram(proba[y == 1], bins=edges)
    return [{"bin": round(float(edges[i]), 2), "legitimate": int(legit[i]), "phishing": int(phish[i])}
            for i in range(bins)]


def _threshold_sweep(y: np.ndarray, proba: np.ndarray) -> list[dict]:
    out = []
    for t in np.linspace(0.05, 0.95, 19):
        m = metrics_at(y, proba, float(t))
        out.append({"threshold": round(float(t), 2), "precision": m["precision"],
                    "recall": m["recall"], "f1": m["f1"]})
    return out


def train(cfg: ClassifierConfig,
          on_event: Callable[[dict], None] = lambda e: None,
          should_stop: Callable[[], bool] = lambda: False) -> ClassifierBundle:
    t0 = time.perf_counter()
    on_event({"type": "stage", "stage": "features", "message": "Extrayendo características léxicas"})
    x_tr_all, y_tr_all, _ = _split_xy("train")
    x_val, y_val, _ = _split_xy("val")
    x_te, y_te, _ = _split_xy("test")

    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.train_size, len(y_tr_all))
    # Stratified subsample of the train split (keeps the class ratio of the split).
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y_tr_all == c),
                   size=int(round(n * (y_tr_all == c).mean())), replace=False)
        for c in (0, 1)])
    rng.shuffle(idx)
    x_tr, y_tr = x_tr_all[idx], y_tr_all[idx]
    on_event({"type": "stage", "stage": "fit",
              "message": f"Entrenando con {len(y_tr)} dominios (train), validando con {len(y_val)}"})

    # warm_start + "balanced" presets would recompute weights per fit; compute them once
    # from the training sample instead (the approach recommended by scikit-learn).
    weights = cfg.class_weight
    if cfg.class_weight == "balanced":
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
        weights = {0: float(w[0]), 1: float(w[1])}
    model = RandomForestClassifier(
        n_estimators=0, max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
        class_weight=weights, random_state=cfg.seed, n_jobs=cfg.n_jobs, warm_start=True)
    history: list[dict] = []
    step = max(1, cfg.n_estimators // 20)
    grown = 0
    while grown < cfg.n_estimators:
        if should_stop():
            on_event({"type": "stage", "stage": "stopped", "message": "Detenido por el usuario"})
            break
        grown = min(cfg.n_estimators, grown + step)
        model.set_params(n_estimators=grown)
        ts = time.perf_counter()
        model.fit(x_tr, y_tr)
        p_val = model.predict_proba(x_val)[:, 1]
        m = metrics_at(y_val, p_val, cfg.threshold)
        point = {"trees": grown, "val_accuracy": m["accuracy"], "val_precision": m["precision"],
                 "val_recall": m["recall"], "val_f1": m["f1"], "val_auc": m["roc_auc"],
                 "train_accuracy": float(accuracy_score(y_tr, model.predict(x_tr))),
                 "step_seconds": time.perf_counter() - ts,
                 "elapsed": time.perf_counter() - t0}
        history.append(point)
        on_event({"type": "metrics", **point})

    if grown == 0:
        raise RuntimeError("Entrenamiento detenido antes de crear ningún árbol")
    cfg.n_estimators = len(model.estimators_)

    on_event({"type": "stage", "stage": "evaluate", "message": "Evaluando en test (no visto)"})
    p_val = model.predict_proba(x_val)[:, 1]
    p_te = model.predict_proba(x_te)[:, 1]
    sub = rng.choice(len(y_val), size=min(4000, len(y_val)), replace=False)
    perm = permutation_importance(model, x_val[sub], y_val[sub], n_repeats=3,
                                  random_state=cfg.seed, n_jobs=cfg.n_jobs, scoring="f1")
    ti = time.perf_counter()
    model.predict_proba(x_te[:1000])
    infer_ms_per_1k = (time.perf_counter() - ti) * 1000

    importance = [
        {"feature": f, "description": features.FEATURE_DESCRIPTIONS[f],
         "impurity": float(model.feature_importances_[i]),
         "permutation": float(perm.importances_mean[i]),
         "permutation_std": float(perm.importances_std[i])}
        for i, f in enumerate(features.FEATURE_NAMES)]
    importance.sort(key=lambda d: d["impurity"], reverse=True)

    metrics = {
        "val": metrics_at(y_val, p_val, cfg.threshold),
        "test": metrics_at(y_te, p_te, cfg.threshold),
        "test_histogram": _histogram(y_te, p_te),
        "test_threshold_sweep": _threshold_sweep(y_te, p_te),
        "feature_importance": importance,
        "train_rows": int(len(y_tr)),
        "train_phishing_ratio": float(y_tr.mean()),
        "train_seconds": time.perf_counter() - t0,
        "inference_ms_per_1000": infer_ms_per_1k,
        "n_trees": cfg.n_estimators,
        "split_note": "train/val/test separated by registered domain (eTLD+1); "
                      "test never used during training",
    }
    bundle = ClassifierBundle(model=model, config=cfg, feature_names=list(features.FEATURE_NAMES),
                              metrics=metrics, history=history)
    on_event({"type": "done", "metrics": {k: metrics[k] for k in ("val", "test", "train_seconds")}})
    return bundle


def rethreshold(bundle: ClassifierBundle, threshold: float) -> dict:
    """Recompute test metrics for another decision threshold without retraining."""
    x_te, y_te, _ = _split_xy("test")
    return metrics_at(y_te, bundle.model.predict_proba(x_te)[:, 1], threshold)


def config_dict(cfg: ClassifierConfig) -> dict:
    return asdict(cfg)


def contributions(bundle: ClassifierBundle, name: str) -> dict:
    """Exact per-feature decomposition of the forest probability for one name.

    For every tree, walk the decision path and attribute the change in the node's
    phishing-class fraction to the feature used by the parent split (Saabas method).
    Averaged over trees:  prior + sum(contributions) == predict_proba (up to float error).
    """
    x = features.matrix([name])
    total = np.zeros(len(features.FEATURE_NAMES))
    prior = 0.0
    for est in bundle.model.estimators_:
        tree = est.tree_
        value = tree.value[:, 0, :]
        frac = value[:, 1] / value.sum(axis=1)
        path = est.decision_path(x).indices  # node ids from root to leaf, in order
        prior += frac[path[0]]
        for parent, child in zip(path[:-1], path[1:]):
            total[tree.feature[parent]] += frac[child] - frac[parent]
    n = len(bundle.model.estimators_)
    total /= n
    prior /= n
    values = features.extract(name)
    return {
        "prior": float(prior),
        "probability": float(prior + total.sum()),
        "contributions": sorted(
            [{"feature": f, "description": features.FEATURE_DESCRIPTIONS[f],
              "value": values[f], "contribution": float(total[i])}
             for i, f in enumerate(features.FEATURE_NAMES)],
            key=lambda d: abs(d["contribution"]), reverse=True),
    }


@lru_cache(maxsize=1)
def class_medians() -> dict:
    """Median of each feature per class on the train split (context for the Inspector)."""
    x, y, _ = _split_xy("train")
    return {f: {"legitimate": float(np.median(x[y == 0, i])), "phishing": float(np.median(x[y == 1, i]))}
            for i, f in enumerate(features.FEATURE_NAMES)}
