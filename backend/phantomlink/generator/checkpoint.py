"""Generator checkpoint (.pt): weights + architecture + tokenizer + training history."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .model import GPTConfig, MiniGPT
from .tokenizer import CharTokenizer

FORMAT = "phantomlink-generator/1"


@dataclass
class GeneratorBundle:
    model: MiniGPT
    tokenizer: CharTokenizer
    model_config: GPTConfig
    train_config: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": FORMAT,
                    "model_config": self.model_config.to_dict(),
                    "state_dict": self.model.state_dict(),
                    "tokenizer": self.tokenizer.state(),
                    "train_config": self.train_config,
                    "history": self.history,
                    "metrics": self.metrics}, path)

    @staticmethod
    def load(path: Path, device: str = "cpu") -> "GeneratorBundle":
        # weights_only=True: the checkpoint contains only tensors and plain Python containers.
        ck = torch.load(path, map_location=device, weights_only=True)
        if ck.get("format") != FORMAT:
            raise ValueError(f"{path} is not a PHANTOMLINK generator checkpoint")
        cfg = GPTConfig(**ck["model_config"])
        model = MiniGPT(cfg)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return GeneratorBundle(model=model, tokenizer=CharTokenizer(**ck["tokenizer"]),
                               model_config=cfg, train_config=ck.get("train_config", {}),
                               history=ck.get("history", []), metrics=ck.get("metrics", {}))
