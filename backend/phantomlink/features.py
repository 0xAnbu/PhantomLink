"""Lexical features computed locally from a normalised domain name (no network, no page content).

Every feature is a pure function of the string, so the exact same transformation runs at
training time, in the Arena and in the Inspector.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

VOWELS = set("aeiou")
# Lexical patterns frequently abused in deceptive hostnames. Presence is counted, never used
# as a rule on its own: the forest decides how much (if anything) it matters.
SUSPICIOUS_TOKENS = (
    "login", "signin", "secure", "account", "verify", "verif", "update", "confirm", "bank",
    "pay", "wallet", "support", "service", "auth", "recover", "unlock", "billing", "web",
    "app", "online", "free", "gift", "bonus", "claim", "help", "cloud", "mail", "office",
    "id", "token", "crypto", "nft", "airdrop",
)

FEATURES: list[tuple[str, str]] = [
    ("length", "Número total de caracteres"),
    ("n_labels", "Número de etiquetas separadas por punto (subdominios + nombre)"),
    ("max_label_len", "Longitud de la etiqueta más larga"),
    ("mean_label_len", "Longitud media de las etiquetas"),
    ("n_digits", "Cantidad de dígitos"),
    ("digit_ratio", "Proporción de dígitos"),
    ("n_hyphens", "Cantidad de guiones"),
    ("hyphen_ratio", "Proporción de guiones"),
    ("entropy", "Entropía de Shannon de los caracteres (bits)"),
    ("distinct_ratio", "Caracteres distintos / longitud"),
    ("vowel_ratio", "Proporción de vocales entre las letras"),
    ("max_consonant_run", "Racha más larga de consonantes"),
    ("max_digit_run", "Racha más larga de dígitos"),
    ("max_repeat_run", "Racha más larga del mismo carácter"),
    ("repeated_bigram_ratio", "Proporción de bigramas repetidos"),
    ("letter_digit_switches", "Cambios letra↔dígito"),
    ("suspicious_tokens", "Nº de patrones léxicos sospechosos presentes"),
    ("has_punycode", "Contiene 'xn--' (IDN)"),
    ("starts_with_digit", "Empieza por dígito"),
]
FEATURE_NAMES = [f for f, _ in FEATURES]
FEATURE_DESCRIPTIONS = dict(FEATURES)

_CONS_RUN = re.compile(r"[bcdfghjklmnpqrstvwxyz]+")
_DIGIT_RUN = re.compile(r"[0-9]+")


def _max_run(pattern: re.Pattern, s: str) -> int:
    return max((len(m) for m in pattern.findall(s)), default=0)


def _max_repeat(s: str) -> int:
    best = cur = 0
    prev = None
    for ch in s:
        cur = cur + 1 if ch == prev else 1
        best = max(best, cur)
        prev = ch
    return best


def extract(name: str) -> dict[str, float]:
    name = name.lower()
    n = max(len(name), 1)
    labels = name.split(".")
    letters = [c for c in name if c.isalpha()]
    counts = Counter(name)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values()) if name else 0.0
    bigrams = [name[i:i + 2] for i in range(len(name) - 1)]
    rep_bigrams = (len(bigrams) - len(set(bigrams))) / len(bigrams) if bigrams else 0.0
    switches = sum(1 for a, b in zip(name, name[1:])
                   if (a.isdigit() and b.isalpha()) or (a.isalpha() and b.isdigit()))
    n_digits = sum(c.isdigit() for c in name)
    n_hyphens = name.count("-")
    return {
        "length": float(len(name)),
        "n_labels": float(len(labels)),
        "max_label_len": float(max(len(l) for l in labels)),
        "mean_label_len": float(np.mean([len(l) for l in labels])),
        "n_digits": float(n_digits),
        "digit_ratio": n_digits / n,
        "n_hyphens": float(n_hyphens),
        "hyphen_ratio": n_hyphens / n,
        "entropy": float(entropy),
        "distinct_ratio": len(counts) / n,
        "vowel_ratio": (sum(c in VOWELS for c in letters) / len(letters)) if letters else 0.0,
        "max_consonant_run": float(_max_run(_CONS_RUN, name)),
        "max_digit_run": float(_max_run(_DIGIT_RUN, name)),
        "max_repeat_run": float(_max_repeat(name)),
        "repeated_bigram_ratio": float(rep_bigrams),
        "letter_digit_switches": float(switches),
        "suspicious_tokens": float(sum(tok in name for tok in SUSPICIOUS_TOKENS)),
        "has_punycode": float("xn--" in name),
        "starts_with_digit": float(name[:1].isdigit()),
    }


def matrix(names) -> np.ndarray:
    """Feature matrix (n_samples, n_features) in FEATURE_NAMES order."""
    rows = [extract(nm) for nm in names]
    return np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=np.float32)
