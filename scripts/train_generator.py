"""Train the Mini-GPT domain generator from the command line and save it as a checkpoint.

    python scripts/train_generator.py                 # -> checkpoints/generator_default.pt
    python scripts/train_generator.py --iterations 500 --n-layer 1 --out runs/generator_tiny.pt
"""
import argparse
from dataclasses import fields
from pathlib import Path

import _common  # noqa: F401
from phantomlink import paths
from phantomlink.generator import GenTrainConfig, generate, train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in fields(GenTrainConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    ap.add_argument("--out", type=Path, default=paths.CHECKPOINT_DIR / "generator_default.pt")
    a = vars(ap.parse_args())
    out = a.pop("out")
    cfg = GenTrainConfig(**a)

    def log(e):
        if e["type"] == "stage":
            print("·", e["message"])
        elif e["type"] == "metrics" and "val_loss" in e:
            print(f"  it={e['iter']:5d}  train={e['train_loss']:.3f}  val={e['val_loss']:.3f}  "
                  f"ppl={e['val_ppl']:.2f}  |g|={e['grad_norm']:.2f}  {e['tokens_per_sec']:.0f} tok/s"
                  f"  phish→ {', '.join(e['samples']['phish'])}")

    bundle = train(cfg, on_event=log)
    bundle.save(out)
    for style in ("legit", "phish"):
        names = [g["name"] for g in generate(bundle.model, bundle.tokenizer, style, 8, seed=7)]
        print(f"{style:6s}", ", ".join(names))
    print(bundle.metrics)
    print(f"Saved {out}")
