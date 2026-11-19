"""AI Arena: the generator produces names, they are validated, the detector judges them.

A generation style ("phish"/"legit") is the *conditioning* requested from the generator, not
ground truth: a "phish-style" string flagged by the detector is agreement between two models,
not a verified phishing site, and "not suspicious" never means "safe".
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from . import data
from .classifier import ClassifierBundle
from .generator import GeneratorBundle, generate


def invalid_reason(name: str) -> str | None:
    if not name:
        return "vacío"
    if len(name) > 253 - len(data.FICTITIOUS_TLD) - 1:
        return "demasiado largo"
    for label in name.split("."):
        if not label:
            return "etiqueta vacía (..)"
        if len(label) > 63:
            return "etiqueta > 63 caracteres"
        if label.startswith("-") or label.endswith("-"):
            return "guion al inicio/fin de etiqueta"
    return None if data.is_valid_name(name) else "carácter no permitido"


@lru_cache(maxsize=1)
def dataset_index() -> dict[str, str]:
    df = data.load_processed()
    return dict(zip(df["name"], df["split"]))


def run(gen: GeneratorBundle, clf: ClassifierBundle, count: int, mix: str, temperature: float,
        top_k: int, max_len: int | None, seed: int, threshold: float) -> dict:
    if mix == "mixed":
        plan = [("phish", count - count // 2), ("legit", count // 2)]
    else:
        plan = [(mix, count)]
    t0 = time.perf_counter()
    generated = []
    for i, (style, n) in enumerate(plan):
        if n:
            # different seed per style so both styles do not share the same random stream
            generated += generate(gen.model, gen.tokenizer, style=style, n=n,
                                  temperature=temperature, top_k=top_k, max_len=max_len,
                                  seed=seed + 7919 * i)
    gen_ms = (time.perf_counter() - t0) * 1000

    index = dataset_index()
    seen: set[str] = set()
    items = []
    for i, g in enumerate(generated):
        reason = invalid_reason(g["name"])
        items.append({"id": i, "name": g["name"],
                      "display": f"{g['name']}.{data.FICTITIOUS_TLD}" if g["name"] else "",
                      "style": g["style"], "valid": reason is None, "invalid_reason": reason,
                      "truncated": g["truncated"], "avg_logprob": g["avg_logprob"],
                      "duplicate": g["name"] in seen, "in_dataset": index.get(g["name"]),
                      "probability": None, "suspicious": None})
        seen.add(g["name"])

    valid = [it for it in items if it["valid"]]
    t1 = time.perf_counter()
    proba = clf.predict_proba([it["name"] for it in valid])
    clf_ms = (time.perf_counter() - t1) * 1000
    for it, p in zip(valid, proba):
        it["probability"] = float(p)
        it["suspicious"] = bool(p >= threshold)

    def style_summary(style: str) -> dict:
        sub = [it for it in valid if it["style"] == style]
        ps = [it["probability"] for it in sub]
        return {"requested": sum(1 for it in items if it["style"] == style), "valid": len(sub),
                "suspicious": sum(it["suspicious"] for it in sub),
                "mean_probability": float(np.mean(ps)) if ps else None}

    summary = {
        "generated": len(items), "valid": len(valid), "invalid": len(items) - len(valid),
        "unique": len(seen), "in_dataset": sum(1 for it in items if it["in_dataset"]),
        "suspicious": sum(it["suspicious"] for it in valid),
        "not_suspicious": sum(not it["suspicious"] for it in valid),
        "mean_probability": float(np.mean(proba)) if len(proba) else None,
        "by_style": {s: style_summary(s) for s, _ in plan},
        "generation_ms": gen_ms, "classification_ms": clf_ms,
    }
    return {"items": items, "summary": summary,
            "settings": {"count": count, "mix": mix, "temperature": temperature, "top_k": top_k,
                         "max_len": max_len, "seed": seed, "threshold": threshold}}
