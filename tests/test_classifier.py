import numpy as np

from phantomlink import classifier


def test_train_metrics_and_contributions(tiny_dataset, tmp_path):
    cfg = classifier.ClassifierConfig(n_estimators=20, max_depth=6, train_size=1000, n_jobs=1)
    events = []
    b = classifier.train(cfg, on_event=events.append)
    m = b.metrics
    assert m["test"]["f1"] > 0.9   # synthetic classes are lexically separable
    c = m["test"]["confusion"]
    assert c["tn"] + c["fp"] + c["fn"] + c["tp"] == m["test"]["n"]
    assert any(e["type"] == "metrics" for e in events)
    assert {d["feature"] for d in m["feature_importance"]}

    # phishing-like synthetic names must score higher (labels interpreted correctly)
    p = b.predict_proba(["secure-garden4821.web", "rivergarden"])
    assert p[0] > p[1]

    # Saabas decomposition is exact for a random forest
    for name in ["secure-garden4821.web", "rivergarden", "x1-y2"]:
        d = classifier.contributions(b, name)
        assert abs(d["probability"] - b.predict_proba([name])[0]) < 1e-6
        assert len(b.tree_votes(name)) == 20

    b.save(tmp_path / "c.joblib")
    b2 = classifier.ClassifierBundle.load(tmp_path / "c.joblib")
    assert np.allclose(b2.predict_proba(["abc"]), b.predict_proba(["abc"]))


def test_threshold_changes_decisions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.6, 0.55, 0.9])
    lo = classifier.metrics_at(y, p, 0.5)
    hi = classifier.metrics_at(y, p, 0.8)
    assert lo["confusion"]["fp"] == 1 and hi["confusion"]["fp"] == 0
    assert hi["recall"] < lo["recall"]
