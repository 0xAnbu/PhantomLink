import pytest
from fastapi.testclient import TestClient

from phantomlink import api, classifier, paths
from phantomlink.generator import GenTrainConfig, train


@pytest.fixture
def client(tiny_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("PHANTOMLINK_SKIP_AUTOLOAD", "1")
    gen = train(GenTrainConfig(n_layer=1, n_head=2, n_embd=32, block_size=32, iterations=60,
                               eval_interval=30, device="cpu"))
    gen.save(paths.CHECKPOINT_DIR / api.DEFAULT_GENERATOR)
    clf = classifier.train(classifier.ClassifierConfig(n_estimators=10, train_size=1000, n_jobs=1))
    clf.save(paths.CHECKPOINT_DIR / api.DEFAULT_CLASSIFIER)
    api.state = api.AppState()
    api.state.load_defaults()
    with TestClient(api.app) as c:
        yield c


def test_status_and_models(client):
    s = client.get("/api/status").json()
    assert s["generator"]["id"] == api.DEFAULT_GENERATOR
    assert s["classifier"]["id"] == api.DEFAULT_CLASSIFIER
    m = client.get("/api/models").json()
    assert m["generator"][0]["active"]


def test_arena_round(client):
    r = client.post("/api/arena", json={"count": 12, "mix": "mixed", "seed": 3}).json()
    s = r["summary"]
    assert s["generated"] == 12 and s["valid"] + s["invalid"] == 12
    assert s["suspicious"] + s["not_suspicious"] == s["valid"]
    for it in r["items"]:
        assert it["display"] == "" or it["display"].endswith(".test")
        if it["valid"]:
            assert 0 <= it["probability"] <= 1
    again = client.post("/api/arena", json={"count": 12, "mix": "mixed", "seed": 3}).json()
    assert [i["name"] for i in again["items"]] == [i["name"] for i in r["items"]]


def test_classify_normalises_user_input(client):
    r = client.post("/api/classify", json={"domains": ["www.Secure-Login.example.com",
                                                       "rivergarden.test", "plainname"]}).json()
    assert [i["name"] for i in r["items"]] == ["secure-login.example", "rivergarden", "plainname"]
    assert client.post("/api/classify", json={"domains": ["bad-.x"]}).status_code == 422


def test_inspectors(client):
    t = client.post("/api/inspect/generator", json={"style": "phish", "seed": 1}).json()
    assert t["steps"] and t["attention"]
    s = client.post("/api/inspect/generator",
                    json={"prefix": "abc", "force_full": True}).json()
    assert s["name"] == "abc"
    c = client.post("/api/inspect/classifier", json={"domain": "secure-login123.web"}).json()
    total = c["prior"] + sum(x["contribution"] for x in c["contributions"])
    assert abs(total - c["probability"]) < 1e-6


def test_config_validation(client):
    bad = client.post("/api/train/generator", json={"n_embd": 30, "n_head": 4})
    assert bad.status_code == 422
    assert client.post("/api/train/classifier", json={"n_estimators": 0}).status_code == 422
    assert client.post("/api/arena", json={"count": 10_000}).status_code == 422


def test_training_lifecycle_single_slot_and_stop(client):
    r = client.post("/api/train/generator", json={"n_layer": 1, "n_head": 2, "n_embd": 16,
                                                  "block_size": 32, "iterations": 5000,
                                                  "eval_interval": 1000, "device": "cpu"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    # a second concurrent training is rejected
    assert client.post("/api/train/classifier", json={}).status_code == 409
    import time
    t0 = time.time()
    while not any(e["type"] == "metrics" for e in api.state.jobs.get(job_id).events):
        assert time.time() - t0 < 30
        time.sleep(0.05)
    assert client.post("/api/train/stop").status_code == 200
    job = api.state.jobs.wait(job_id, timeout=60)
    assert job.status == "stopped"
    assert job.result["saved"].startswith("generator_")
    # SSE replay ends with an "end" event
    with client.stream("GET", f"/api/jobs/{job_id}/events") as s:
        body = "".join(s.iter_text())
    assert "event: end" in body and "event: metrics" in body
    assert api.state.generator_id == job.result["saved"]


def test_challenge_uses_test_split_with_labels(client, tiny_dataset):
    r = client.get("/api/challenge", params={"n": 10, "seed": 1}).json()
    test_names = set(tiny_dataset[tiny_dataset.split == "test"]["name"])
    assert len(r["items"]) == 10
    assert all(i["name"] in test_names for i in r["items"])
    assert sum(i["is_phishing"] for i in r["items"]) == 5
