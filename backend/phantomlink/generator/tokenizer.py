"""Character tokenizer with style-conditioning tokens.

A training sequence is:   <style> c1 c2 ... cn <eos>
where <style> is <legit> or <phish>. The style token is the only conditioning signal.
"""

from __future__ import annotations

from ..data import ALPHABET

PAD, LEGIT, PHISH, EOS = "<pad>", "<legit>", "<phish>", "<eos>"
SPECIALS = [PAD, LEGIT, PHISH, EOS]
STYLES = {"legit": LEGIT, "phish": PHISH}


class CharTokenizer:
    def __init__(self, alphabet: str = ALPHABET):
        self.alphabet = alphabet
        self.itos = SPECIALS + list(alphabet)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.eos_id = self.stoi[EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def style_id(self, style: str) -> int:
        return self.stoi[STYLES[style]]

    def encode(self, name: str, style: str) -> list[int]:
        return [self.style_id(style)] + [self.stoi[c] for c in name] + [self.eos_id]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if self.itos[i] not in SPECIALS)

    def token_str(self, i: int) -> str:
        return self.itos[i]

    def state(self) -> dict:
        return {"alphabet": self.alphabet}
