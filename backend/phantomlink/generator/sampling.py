"""Autoregressive sampling and step-by-step inspection.

Everything returned here is read from the model's tensors: logits, softmax probabilities,
attention weights and residual-stream activations. Nothing is approximated for display.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .model import MiniGPT
from .tokenizer import CharTokenizer


def _sampling_probs(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Distribution actually sampled from: temperature scaling then top-k truncation."""
    if temperature <= 0:  # greedy
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return probs
    scaled = logits / temperature
    if 0 < top_k < scaled.size(-1):
        kth = torch.topk(scaled, top_k, dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    return F.softmax(scaled, dim=-1)


def _forbid_specials(logits: torch.Tensor, tok: CharTokenizer, allow_eos: bool) -> torch.Tensor:
    """The model may never emit <pad>/<style> tokens; <eos> is blocked on the first step."""
    logits = logits.clone()
    for s in ("<pad>", "<legit>", "<phish>"):
        logits[..., tok.stoi[s]] = float("-inf")
    if not allow_eos:
        logits[..., tok.eos_id] = float("-inf")
    return logits


@torch.no_grad()
def generate(model: MiniGPT, tok: CharTokenizer, style: str, n: int, temperature: float = 0.8,
             top_k: int = 20, max_len: int | None = None, seed: int = 42,
             device: torch.device | str = "cpu") -> list[dict]:
    """Batched generation of ``n`` names in one style."""
    was_training = model.training
    model.eval()
    block = model.cfg.block_size
    max_len = min(max_len or block - 1, block - 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.full((n, 1), tok.style_id(style), dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    logp_sum = torch.zeros(n, device=device)
    lengths = torch.zeros(n, dtype=torch.long, device=device)
    for step in range(max_len + 1):
        logits, _ = model(idx)
        logits = _forbid_specials(logits[:, -1, :], tok, allow_eos=step > 0)
        if step == max_len:  # force termination at the length limit
            nxt = torch.full((n,), tok.eos_id, dtype=torch.long, device=device)
        else:
            probs = _sampling_probs(logits, temperature, top_k)
            nxt = torch.multinomial(probs.cpu(), 1, generator=g).squeeze(1).to(device)
        model_logp = F.log_softmax(logits, dim=-1).gather(1, nxt[:, None]).squeeze(1)
        active = ~done
        logp_sum += torch.where(active & torch.isfinite(model_logp), model_logp, 0.0)
        lengths += active.long()
        nxt = torch.where(done, torch.full_like(nxt, tok.pad_id), nxt)
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        done |= nxt == tok.eos_id
        if bool(done.all()):
            break
    model.train(was_training)
    out = []
    for i in range(n):
        ids = idx[i, 1:].tolist()
        hit_limit = tok.eos_id not in ids or ids.index(tok.eos_id) >= max_len
        out.append({"name": tok.decode(ids), "style": style,
                    "avg_logprob": float(logp_sum[i] / max(int(lengths[i]), 1)),
                    "truncated": bool(hit_limit)})
    return out


def _entropy(p: torch.Tensor) -> float:
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


@torch.no_grad()
def trace(model: MiniGPT, tok: CharTokenizer, style: str, temperature: float = 0.8,
          top_k: int = 20, max_len: int | None = None, seed: int = 42, prefix: str = "",
          force_full: bool = False, n_candidates: int = 8) -> dict:
    """Generate (or score, with ``force_full``) one name and expose every internal step.

    ``prefix`` characters are forced; the model then continues on its own. With
    ``force_full`` the whole prefix is scored and terminated with <eos> (teacher forcing).
    """
    model.eval()
    block = model.cfg.block_size
    max_len = min(max_len or block - 1, block - 1)
    for ch in prefix:
        if ch not in tok.stoi:
            raise ValueError(f"character {ch!r} is not in the vocabulary")
    prefix = prefix[:max_len]
    g = torch.Generator().manual_seed(seed)
    ids = [tok.style_id(style)]
    forced_flags: list[bool] = []
    t0 = time.perf_counter()
    for step in range(max_len + 1):
        if step < len(prefix):
            nxt, forced = tok.stoi[prefix[step]], True
        elif force_full or step == max_len:
            nxt, forced = tok.eos_id, True
        else:
            logits, _ = model(torch.tensor([ids]))
            logits = _forbid_specials(logits[0, -1], tok, allow_eos=step > 0)
            probs = _sampling_probs(logits, temperature, top_k)
            nxt, forced = int(torch.multinomial(probs, 1, generator=g)), False
        ids.append(nxt)
        forced_flags.append(forced)
        if nxt == tok.eos_id:
            break
    gen_ms = (time.perf_counter() - t0) * 1000

    # One causal forward pass over the final sequence gives, at every position, exactly the
    # distribution the model had when that next token was chosen, plus attention/activations.
    x = torch.tensor([ids[:-1]])
    logits, _, internals = model(x, inspect=True)
    steps = []
    total_logp = 0.0
    for pos in range(x.size(1)):
        raw = logits[0, pos]
        masked = _forbid_specials(raw, tok, allow_eos=pos > 0)
        p_model = F.softmax(masked, dim=-1)
        p_sample = _sampling_probs(masked, temperature, top_k)
        chosen = ids[pos + 1]
        top = torch.topk(p_model, n_candidates)
        total_logp += math.log(max(float(p_model[chosen]), 1e-12))
        steps.append({
            "position": pos,
            "context": tok.decode(ids[1:pos + 1]),
            "chosen": tok.token_str(chosen),
            "forced": forced_flags[pos],
            "p_model": float(p_model[chosen]),
            "p_sample": float(p_sample[chosen]),
            "entropy_model": _entropy(p_model),
            "entropy_sample": _entropy(p_sample),
            "candidates": [{"token": tok.token_str(int(i)), "p_model": float(p),
                            "p_sample": float(p_sample[int(i)])}
                           for p, i in zip(top.values, top.indices)],
        })
    tokens = [tok.token_str(i) for i in ids[:-1]]
    attention = [[[[round(float(v), 4) for v in row] for row in head] for head in layer[0]]
                 for layer in internals["attention"]]
    activations = [[[round(float(v), 3) for v in vec] for vec in layer[0]]
                   for layer in internals["residual"]]
    n_tok = len(steps)
    return {
        "name": tok.decode(ids),
        "style": style,
        "tokens": tokens,
        "steps": steps,
        "attention": attention,          # [layer][head][query][key]
        "activations": activations,      # [embed, block1, ...][position][dim]
        "activation_labels": ["embeddings"] + [f"block {i + 1}" for i in range(len(attention))],
        "logprob": total_logp,
        "perplexity": math.exp(-total_logp / max(n_tok, 1)),
        "generation_ms": gen_ms,
        "settings": {"temperature": temperature, "top_k": top_k, "seed": seed,
                     "max_len": max_len, "prefix": prefix, "force_full": force_full},
    }
