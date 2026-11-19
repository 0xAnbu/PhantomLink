"""Training loop for the domain generator with real-time telemetry."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import psutil
import torch

from .. import data
from .checkpoint import GeneratorBundle
from .model import GPTConfig, MiniGPT
from .sampling import generate
from .tokenizer import CharTokenizer


@dataclass
class GenTrainConfig:
    # architecture (requires retraining)
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    block_size: int = 64
    dropout: float = 0.1
    # optimisation
    learning_rate: float = 1e-3
    batch_size: int = 32
    iterations: int = 1000
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0          # 0 disables clipping
    seed: int = 42
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    device: str = "auto"            # auto | cpu | cuda


def resolve_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def build_corpus(tok: CharTokenizer, block_size: int, split: str) -> tuple[torch.Tensor, dict]:
    """Padded (N, block_size + 1) tensor of encoded names from one split."""
    df = data.load_processed()
    part = df[df.split == split]
    max_chars = block_size - 1  # style + chars + eos must fit in block_size + 1 tokens
    keep = part[part["name"].str.len() <= max_chars]
    seqs = np.full((len(keep), block_size + 1), tok.pad_id, dtype=np.int64)
    for i, (name, ph) in enumerate(zip(keep["name"], keep["is_phishing"])):
        ids = tok.encode(name, "phish" if ph else "legit")
        seqs[i, :len(ids)] = ids
    info = {"split": split, "sequences": int(len(keep)), "dropped_too_long": int(len(part) - len(keep)),
            "phish": int(keep["is_phishing"].sum()), "legit": int((keep["is_phishing"] == 0).sum())}
    return torch.from_numpy(seqs), info


def lr_at(it: int, cfg: GenTrainConfig) -> float:
    warm = max(1, int(cfg.iterations * cfg.warmup_frac))
    if it < warm:
        return cfg.learning_rate * (it + 1) / warm
    progress = (it - warm) / max(1, cfg.iterations - warm)
    return cfg.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _memory(device: torch.device) -> dict:
    mem = {"rss_mb": psutil.Process(os.getpid()).memory_info().rss / 2**20}
    if device.type == "cuda":
        mem["cuda_mb"] = torch.cuda.memory_allocated(device) / 2**20
    return mem


@torch.no_grad()
def _estimate(model: MiniGPT, corpus: torch.Tensor, cfg: GenTrainConfig, device, gen) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        ix = torch.randint(len(corpus), (cfg.batch_size,), generator=gen)
        batch = corpus[ix].to(device)
        _, loss = model(batch[:, :-1], batch[:, 1:], ignore_index=0)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(cfg: GenTrainConfig,
          on_event: Callable[[dict], None] = lambda e: None,
          should_stop: Callable[[], bool] = lambda: False) -> GeneratorBundle:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    tok = CharTokenizer()
    gcfg = GPTConfig(vocab_size=tok.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
                     n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout)
    model = MiniGPT(gcfg).to(device)
    on_event({"type": "stage", "stage": "corpus", "message": "Construyendo corpus de dominios"})
    train_c, train_info = build_corpus(tok, cfg.block_size, "train")
    val_c, val_info = build_corpus(tok, cfg.block_size, "val")
    on_event({"type": "stage", "stage": "model", "params": model.num_params(), "device": str(device),
              "train_corpus": train_info, "val_corpus": val_info,
              "message": f"MiniGPT con {model.num_params():,} parámetros en {device}"})

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.learning_rate, betas=(0.9, 0.99))
    gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 1)
    history: list[dict] = []
    t_start = time.perf_counter()
    window_tokens, window_t = 0, time.perf_counter()
    model.train()
    it = 0
    stopped = False
    last_val = None
    diverged = False
    for it in range(cfg.iterations):
        if should_stop():
            stopped = True
            break
        lr = lr_at(it, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        t_iter = time.perf_counter()
        ix = torch.randint(len(train_c), (cfg.batch_size,), generator=gen)
        batch = train_c[ix].to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        _, loss = model(x, y, ignore_index=tok.pad_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Real L2 norm of all gradients (before clipping).
        clip = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        loss_v = float(loss.item())
        window_tokens += int((y != tok.pad_id).sum())
        ms_iter = (time.perf_counter() - t_iter) * 1000

        if not math.isfinite(loss_v):
            diverged = True
            on_event({"type": "stage", "stage": "diverged",
                      "message": f"La loss ha divergido (={loss_v}) en la iteración {it}"})
            break

        if it % cfg.log_interval == 0 or it == cfg.iterations - 1:
            now = time.perf_counter()
            point = {"iter": it, "train_loss": loss_v, "lr": lr, "grad_norm": grad_norm,
                     "tokens_per_sec": window_tokens / max(now - window_t, 1e-9),
                     "ms_per_iter": ms_iter, "elapsed": now - t_start, **_memory(device)}
            window_tokens, window_t = 0, now
            if it % cfg.eval_interval == 0 or it == cfg.iterations - 1:
                tr = _estimate(model, train_c, cfg, device, eval_gen)
                va = _estimate(model, val_c, cfg, device, eval_gen)
                last_val = va
                point.update({"eval_train_loss": tr, "val_loss": va, "val_ppl": math.exp(va)})
                model.eval()
                samples = {s: generate(model, tok, style=s, n=3, temperature=0.8, top_k=20,
                                       seed=cfg.seed + it, device=device)
                           for s in ("legit", "phish")}
                model.train()
                point["samples"] = {s: [g["name"] for g in v] for s, v in samples.items()}
            history.append(point)
            on_event({"type": "metrics", **point})

    model.eval()
    final_val = _estimate(model, val_c, cfg, device, eval_gen) if not diverged else float("nan")
    metrics = {"final_val_loss": final_val,
               "final_val_ppl": math.exp(final_val) if math.isfinite(final_val) else None,
               "best_val_loss": min((h["val_loss"] for h in history if "val_loss" in h), default=None),
               "iterations_done": it + (0 if stopped or diverged else 1),
               "stopped": stopped, "diverged": diverged,
               "train_seconds": time.perf_counter() - t_start,
               "params": model.num_params(), "device": str(device),
               "train_corpus": train_info, "val_corpus": val_info}
    bundle = GeneratorBundle(model=model.cpu(), tokenizer=tok, model_config=gcfg,
                             train_config=asdict(cfg), history=history, metrics=metrics)
    on_event({"type": "done", "metrics": metrics})
    return bundle
