import math

import torch

from phantomlink.generator import (CharTokenizer, GeneratorBundle, GenTrainConfig, GPTConfig,
                                   MiniGPT, generate, trace, train)


def _model(**kw):
    tok = CharTokenizer()
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=4, n_embd=32,
                    dropout=0.0, **kw)
    return MiniGPT(cfg).eval(), tok


def test_tokenizer_roundtrip():
    tok = CharTokenizer()
    ids = tok.encode("secure-login.web", "phish")
    assert ids[0] == tok.style_id("phish") and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "secure-login.web"


def test_attention_is_causal_and_normalised():
    model, tok = _model()
    x = torch.tensor([tok.encode("abcdef", "legit")[:-1]])
    logits, _, inner = model(x, inspect=True)
    att = inner["attention"][0][0]                     # (heads, t, t)
    assert torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=1e-5)
    assert torch.all(att.triu(1) == 0)
    # changing a future token must not change earlier predictions
    y = x.clone()
    y[0, -1] = tok.stoi["z"]
    logits2, _ = model(y)
    assert torch.allclose(logits[0, :-1], logits2[0, :-1], atol=1e-6)


def test_generation_is_reproducible_with_seed():
    model, tok = _model()
    a = generate(model, tok, "phish", n=5, seed=123, max_len=12)
    b = generate(model, tok, "phish", n=5, seed=123, max_len=12)
    c = generate(model, tok, "phish", n=5, seed=124, max_len=12)
    assert [g["name"] for g in a] == [g["name"] for g in b]
    assert [g["name"] for g in a] != [g["name"] for g in c]
    assert all(len(g["name"]) <= 12 for g in a)


def test_trace_matches_model_probabilities():
    model, tok = _model()
    t = trace(model, tok, "legit", temperature=1.0, top_k=0, seed=5, max_len=10)
    assert len(t["steps"]) == len(t["tokens"])
    for s in t["steps"]:
        assert 0 < s["p_model"] <= 1
        assert abs(sum(c["p_model"] for c in s["candidates"]) - 1) < 1 or len(s["candidates"]) < 42
    # with T=1 and no top-k the sampling distribution equals the model distribution
    assert all(abs(s["p_model"] - s["p_sample"]) < 1e-5 for s in t["steps"])
    assert len(t["attention"]) == 2 and len(t["attention"][0]) == 4
    scored = trace(model, tok, "legit", prefix="abc", force_full=True)
    assert scored["name"] == "abc" and all(s["forced"] for s in scored["steps"])
    assert math.isfinite(scored["perplexity"])


def test_checkpoint_roundtrip(tmp_path):
    model, tok = _model()
    b = GeneratorBundle(model=model, tokenizer=tok, model_config=model.cfg)
    b.save(tmp_path / "g.pt")
    b2 = GeneratorBundle.load(tmp_path / "g.pt")
    x = torch.tensor([[1, 5, 6, 7]])
    assert torch.allclose(model(x)[0], b2.model(x)[0])


def test_training_reduces_loss(tiny_dataset):
    cfg = GenTrainConfig(n_layer=1, n_head=2, n_embd=32, block_size=32, dropout=0.0,
                         iterations=120, eval_interval=40, log_interval=10, batch_size=32,
                         learning_rate=3e-3, seed=1, device="cpu")
    events = []
    b = train(cfg, on_event=events.append)
    evals = [e["val_loss"] for e in events if e.get("val_loss") is not None]
    assert evals[-1] < evals[0]
    assert b.metrics["iterations_done"] == 120
    metric_keys = {"train_loss", "lr", "grad_norm", "tokens_per_sec", "ms_per_iter", "rss_mb"}
    assert metric_keys <= set(next(e for e in events if e["type"] == "metrics"))


def test_training_can_be_stopped(tiny_dataset):
    cfg = GenTrainConfig(n_layer=1, n_head=2, n_embd=16, block_size=32, iterations=1000,
                         eval_interval=500, device="cpu")
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 5

    b = train(cfg, should_stop=stop)
    assert b.metrics["stopped"] and b.metrics["iterations_done"] == 5
