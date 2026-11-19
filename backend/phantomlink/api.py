"""FastAPI backend: training, status, inference and experiments.

Run:  uvicorn phantomlink.api:app --app-dir backend --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from . import __version__, arena, classifier, data, paths, presets
from .classifier import ClassifierBundle, ClassifierConfig
from .generator import GeneratorBundle, GenTrainConfig, trace
from .generator import train as train_generator
from .jobs import JobBusy, JobManager, jsonable

torch.set_num_threads(int(os.environ.get("PHANTOMLINK_THREADS", min(4, os.cpu_count() or 1))))

DEFAULT_GENERATOR = "generator_default.pt"
DEFAULT_CLASSIFIER = "classifier_default.joblib"
MAX_PARAMS = 2_000_000
_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


# ----------------------------------------------------------------------------- state
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.generator: GeneratorBundle | None = None
        self.generator_id: str | None = None
        self.classifier: ClassifierBundle | None = None
        self.classifier_id: str | None = None
        self.jobs = JobManager()

    @staticmethod
    def _dirs():
        return [("shipped", paths.CHECKPOINT_DIR), ("run", paths.RUNS_DIR)]

    def list_checkpoints(self, kind: str) -> list[dict]:
        ext = ".pt" if kind == "generator" else ".joblib"
        out = []
        for origin, d in self._dirs():
            if d.exists():
                for p in sorted(d.glob(f"{kind}_*{ext}")):
                    out.append({"id": p.name, "origin": origin, "size_kb": p.stat().st_size // 1024,
                                "modified": p.stat().st_mtime,
                                "active": p.name == (self.generator_id if kind == "generator"
                                                     else self.classifier_id)})
        return out

    def resolve(self, kind: str, ckpt_id: str) -> Path:
        if not _ID.match(ckpt_id):
            raise HTTPException(400, "identificador de checkpoint no válido")
        for item in self.list_checkpoints(kind):
            if item["id"] == ckpt_id:
                base = paths.CHECKPOINT_DIR if item["origin"] == "shipped" else paths.RUNS_DIR
                return base / ckpt_id
        raise HTTPException(404, f"checkpoint {ckpt_id} no encontrado")

    def activate(self, kind: str, ckpt_id: str) -> None:
        path = self.resolve(kind, ckpt_id)
        if kind == "generator":
            bundle = GeneratorBundle.load(path)
            with self.lock:
                self.generator, self.generator_id = bundle, ckpt_id
        else:
            bundle = ClassifierBundle.load(path)
            with self.lock:
                self.classifier, self.classifier_id = bundle, ckpt_id

    def load_defaults(self) -> None:
        for kind, default in (("generator", DEFAULT_GENERATOR), ("classifier", DEFAULT_CLASSIFIER)):
            ids = [c["id"] for c in self.list_checkpoints(kind)]
            pick = default if default in ids else (ids[-1] if ids else None)
            if pick:
                self.activate(kind, pick)

    def need_generator(self) -> GeneratorBundle:
        if self.generator is None:
            raise HTTPException(503, "no hay generador cargado: entrena uno o ejecuta "
                                     "scripts/train_generator.py")
        return self.generator

    def need_classifier(self) -> ClassifierBundle:
        if self.classifier is None:
            raise HTTPException(503, "no hay clasificador cargado: entrena uno o ejecuta "
                                     "scripts/train_classifier.py")
        return self.classifier


state = AppState()


# ----------------------------------------------------------------------------- schemas
class GenTrainRequest(BaseModel):
    n_layer: int = Field(2, ge=1, le=6)
    n_head: int = Field(4, ge=1, le=8)
    n_embd: int = Field(64, ge=16, le=256)
    block_size: int = Field(64, ge=16, le=64)
    dropout: float = Field(0.1, ge=0.0, le=0.5)
    learning_rate: float = Field(1e-3, gt=0.0, le=1.0)
    batch_size: int = Field(32, ge=4, le=256)
    iterations: int = Field(1000, ge=10, le=20000)
    weight_decay: float = Field(0.01, ge=0.0, le=0.5)
    warmup_frac: float = Field(0.05, ge=0.0, le=0.5)
    grad_clip: float = Field(1.0, ge=0.0, le=10.0)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    eval_interval: int = Field(100, ge=10, le=5000)
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @model_validator(mode="after")
    def _check(self):
        if self.n_embd % self.n_head:
            raise ValueError("n_embd debe ser divisible por n_head")
        approx = 12 * self.n_layer * self.n_embd ** 2 + (42 + self.block_size) * self.n_embd
        if approx > MAX_PARAMS:
            raise ValueError(f"modelo demasiado grande (~{approx:,} parámetros > {MAX_PARAMS:,})")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA no está disponible en este equipo")
        return self


class ClfTrainRequest(BaseModel):
    n_estimators: int = Field(100, ge=1, le=500)
    max_depth: int | None = Field(12, ge=1, le=40)
    min_samples_leaf: int = Field(2, ge=1, le=200)
    class_weight: Literal["balanced"] | None = "balanced"
    threshold: float = Field(0.5, ge=0.01, le=0.99)
    train_size: int = Field(30000, ge=1000, le=200_000)
    seed: int = Field(42, ge=0, le=2**31 - 1)


class SamplingParams(BaseModel):
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(20, ge=0, le=64)
    max_len: int | None = Field(None, ge=1, le=63)
    seed: int = Field(42, ge=0, le=2**31 - 1)


class GenerateRequest(SamplingParams):
    style: Literal["legit", "phish"] = "phish"
    n: int = Field(10, ge=1, le=200)


class ArenaRequest(SamplingParams):
    count: int = Field(20, ge=1, le=200)
    mix: Literal["mixed", "phish", "legit"] = "mixed"
    threshold: float | None = Field(None, ge=0.01, le=0.99)


class TraceRequest(SamplingParams):
    style: Literal["legit", "phish"] = "phish"
    prefix: str = Field("", max_length=63)
    force_full: bool = False


class ClassifyRequest(BaseModel):
    domains: list[str] = Field(..., min_length=1, max_length=200)
    threshold: float | None = Field(None, ge=0.01, le=0.99)


class ActivateRequest(BaseModel):
    kind: Literal["generator", "classifier"]
    id: str = Field(..., max_length=128)


# ----------------------------------------------------------------------------- helpers
def user_domain_to_name(raw: str) -> str:
    """Turn user input into the representation the detector was trained on.

    Accepts a bare name ("secure-login"), a fictitious "x.test" or a full host ("www.x.com").
    Purely string processing: nothing is resolved or contacted.
    """
    s = raw.strip().lower()
    s = re.sub(r"^[a-z]+://", "", s).split("/")[0].rstrip(".")
    if s.endswith("." + data.FICTITIOUS_TLD):
        s = s[: -len(data.FICTITIOUS_TLD) - 1]
    norm = data.normalize_host(s) if "." in s else None
    name = norm.name if norm else (s[4:] if s.startswith("www.") else s)
    if not data.is_valid_name(name):
        raise HTTPException(422, f"'{raw}' no es un nombre de dominio sintácticamente válido")
    return name


def _bundle_summary_generator(b: GeneratorBundle | None, ckpt_id: str | None) -> dict | None:
    if b is None:
        return None
    return {"id": ckpt_id, "params": b.model.num_params(), "config": b.model_config.to_dict(),
            "train_config": b.train_config, "metrics": b.metrics}


def _bundle_summary_classifier(b: ClassifierBundle | None, ckpt_id: str | None) -> dict | None:
    if b is None:
        return None
    m = b.metrics
    return {"id": ckpt_id, "config": asdict(b.config),
            "metrics": {k: m.get(k) for k in ("val", "test", "train_rows", "train_seconds",
                                              "inference_ms_per_1000", "n_trees")}}


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# ----------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("PHANTOMLINK_SKIP_AUTOLOAD") != "1":
        state.load_defaults()
    yield


app = FastAPI(title="PHANTOMLINK", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/status")
def status():
    report = json.loads(paths.SPLIT_REPORT.read_text()) if paths.SPLIT_REPORT.exists() else None
    job = state.jobs.current
    return jsonable({
        "version": __version__,
        "device": {"cuda_available": torch.cuda.is_available(), "torch": torch.__version__,
                   "threads": torch.get_num_threads()},
        "generator": _bundle_summary_generator(state.generator, state.generator_id),
        "classifier": _bundle_summary_classifier(state.classifier, state.classifier_id),
        "job": job.public() if job else None,
        "dataset": report,
    })


@app.get("/api/presets")
def get_presets():
    return {"generator": presets.GENERATOR_PRESETS, "classifier": presets.CLASSIFIER_PRESETS,
            "defaults": {"generator": asdict(GenTrainConfig()),
                         "classifier": asdict(ClassifierConfig())}}


@app.get("/api/models")
def list_models():
    return {"generator": state.list_checkpoints("generator"),
            "classifier": state.list_checkpoints("classifier")}


@app.post("/api/models/activate")
def activate_model(req: ActivateRequest):
    if state.jobs.busy():
        raise HTTPException(409, "no se puede cambiar de modelo durante un entrenamiento")
    state.activate(req.kind, req.id)
    return list_models()


# --- training ---------------------------------------------------------------------------
@app.post("/api/train/generator", status_code=202)
def start_generator_training(req: GenTrainRequest):
    cfg = GenTrainConfig(**req.model_dump())

    def job(emit, should_stop):
        bundle = train_generator(cfg, on_event=emit, should_stop=should_stop)
        if bundle.metrics.get("diverged"):
            emit({"type": "stage", "stage": "not_saved",
                  "message": "Pesos no finitos: no se guarda; el generador activo no cambia"})
            return jsonable({"saved": None, "metrics": bundle.metrics})
        ckpt_id = f"generator_{_timestamp()}.pt"
        bundle.save(paths.RUNS_DIR / ckpt_id)
        with state.lock:
            state.generator, state.generator_id = bundle, ckpt_id
        emit({"type": "stage", "stage": "saved", "checkpoint": ckpt_id,
              "message": f"Guardado y activado {ckpt_id}"})
        return jsonable({"saved": ckpt_id, "metrics": bundle.metrics})

    return _start("generator", asdict(cfg), job)


@app.post("/api/train/classifier", status_code=202)
def start_classifier_training(req: ClfTrainRequest):
    cfg = ClassifierConfig(**req.model_dump())

    def job(emit, should_stop):
        bundle = classifier.train(cfg, on_event=emit, should_stop=should_stop)
        ckpt_id = f"classifier_{_timestamp()}.joblib"
        bundle.save(paths.RUNS_DIR / ckpt_id)
        with state.lock:
            state.classifier, state.classifier_id = bundle, ckpt_id
        emit({"type": "stage", "stage": "saved", "checkpoint": ckpt_id,
              "message": f"Guardado y activado {ckpt_id}"})
        return jsonable({"saved": ckpt_id, "metrics": {k: bundle.metrics[k] for k in ("val", "test")}})

    return _start("classifier", asdict(cfg), job)


def _start(kind, cfg, fn):
    try:
        job = state.jobs.start(kind, cfg, fn)
    except JobBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job_id": job.id, "kind": kind}


@app.post("/api/train/stop")
def stop_training():
    job = state.jobs.stop()
    if not job:
        raise HTTPException(409, "no hay ningún entrenamiento en curso")
    return {"job_id": job.id, "stopping": True}


@app.get("/api/jobs/current")
def current_job():
    job = state.jobs.current
    return jsonable(job.public()) if job else None


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, events: bool = False):
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "trabajo no encontrado")
    return jsonable(job.public(with_events=events))


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, since: int = 0):
    """Server-Sent Events stream of a job's telemetry (replays from ``since``)."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "trabajo no encontrado")

    async def stream():
        # EventSource reconnects send Last-Event-ID: resume after it instead of replaying.
        last_id = request.headers.get("last-event-id")
        cursor = int(last_id) + 1 if last_id and last_id.isdigit() else max(0, since)
        last_ping = time.time()
        while True:
            if await request.is_disconnected():
                return
            while cursor < len(job.events):
                ev = job.events[cursor]
                cursor += 1
                yield f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps(jsonable(ev))}\n\n"
                if ev["type"] == "end":
                    return
            if time.time() - last_ping > 10:
                last_ping = time.time()
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.2)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- inference ----------------------------------------------------------------------------
def _check_len(gen: GeneratorBundle, max_len: int | None) -> None:
    if max_len and max_len > gen.model_config.block_size - 1:
        raise HTTPException(422, f"max_len no puede superar {gen.model_config.block_size - 1} "
                                 "para el generador activo")


@app.post("/api/generate")
def generate_domains(req: GenerateRequest):
    from .generator import generate
    gen = state.need_generator()
    _check_len(gen, req.max_len)
    t0 = time.perf_counter()
    out = generate(gen.model, gen.tokenizer, style=req.style, n=req.n,
                   temperature=req.temperature, top_k=req.top_k, max_len=req.max_len, seed=req.seed)
    for g in out:
        g["valid"] = arena.invalid_reason(g["name"]) is None
    return {"items": out, "ms": (time.perf_counter() - t0) * 1000}


@app.post("/api/classify")
def classify(req: ClassifyRequest):
    clf = state.need_classifier()
    names = [user_domain_to_name(d) for d in req.domains]
    thr = req.threshold if req.threshold is not None else clf.threshold
    t0 = time.perf_counter()
    proba = clf.predict_proba(names)
    ms = (time.perf_counter() - t0) * 1000
    return {"items": [{"input": d, "name": n, "probability": float(p), "suspicious": bool(p >= thr)}
                      for d, n, p in zip(req.domains, names, proba)],
            "threshold": thr, "ms": ms}


@app.post("/api/arena")
def run_arena(req: ArenaRequest):
    gen, clf = state.need_generator(), state.need_classifier()
    _check_len(gen, req.max_len)
    thr = req.threshold if req.threshold is not None else clf.threshold
    result = arena.run(gen, clf, count=req.count, mix=req.mix, temperature=req.temperature,
                       top_k=req.top_k, max_len=req.max_len, seed=req.seed, threshold=thr)
    result["models"] = {"generator": state.generator_id, "classifier": state.classifier_id}
    return jsonable(result)


@app.get("/api/challenge")
def challenge(n: int = 10, seed: int = 0):
    """Human vs AI: random held-out *test* names with their real dataset label.

    Unlike generated names, these have ground truth, so both the visitor's guesses and the
    detector's decisions can be scored. The TLD is shown as .test like everywhere else.
    """
    if not 1 <= n <= 50:
        raise HTTPException(422, "n debe estar entre 1 y 50")
    clf = state.need_classifier()
    df = data.load_processed()
    test = df[df.split == "test"]
    # balanced sample so guessing "always legitimate" does not win
    per = [test[test.is_phishing == c].sample(n=(n + 1 - c) // 2 if c == 0 else n // 2,
                                              random_state=seed) for c in (0, 1)]
    sample = (pd.concat(per).sample(frac=1, random_state=seed) if n > 1 else per[0])
    proba = clf.predict_proba(sample["name"].tolist())
    return {"threshold": clf.threshold, "items": [
        {"name": nm, "display": f"{nm}.{data.FICTITIOUS_TLD}", "is_phishing": bool(lbl),
         "probability": float(p), "suspicious": bool(p >= clf.threshold)}
        for nm, lbl, p in zip(sample["name"], sample["is_phishing"], proba)]}


@app.post("/api/inspect/generator")
def inspect_generator(req: TraceRequest):
    gen = state.need_generator()
    _check_len(gen, req.max_len)
    prefix = req.prefix.strip().lower()
    if req.force_full and not prefix:
        raise HTTPException(422, "para puntuar un dominio hay que indicar el texto")
    try:
        t = trace(gen.model, gen.tokenizer, style=req.style, temperature=req.temperature,
                  top_k=req.top_k, max_len=req.max_len, seed=req.seed, prefix=prefix,
                  force_full=req.force_full)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    t["valid"] = arena.invalid_reason(t["name"]) is None
    return jsonable(t)


class InspectClassifierRequest(BaseModel):
    domain: str = Field(..., min_length=1, max_length=253)


@app.post("/api/inspect/classifier")
def inspect_classifier(req: InspectClassifierRequest):
    clf = state.need_classifier()
    name = user_domain_to_name(req.domain)
    t0 = time.perf_counter()
    contrib = classifier.contributions(clf, name)
    votes = clf.tree_votes(name)
    ms = (time.perf_counter() - t0) * 1000
    medians = classifier.class_medians()
    for c in contrib["contributions"]:
        c["median"] = medians[c["feature"]]
    return jsonable({"input": req.domain, "name": name, "threshold": clf.threshold,
                     "suspicious": contrib["probability"] >= clf.threshold,
                     "tree_votes": votes, "ms": ms, **contrib})


@app.get("/api/classifier/report")
def classifier_report(threshold: float | None = None):
    clf = state.need_classifier()
    m = dict(clf.metrics)
    if threshold is not None:
        if not 0.01 <= threshold <= 0.99:
            raise HTTPException(422, "umbral fuera de rango [0.01, 0.99]")
        m["test_at_threshold"] = classifier.rethreshold(clf, threshold)
    return jsonable({"id": state.classifier_id, "config": asdict(clf.config), "metrics": m,
                     "history": clf.history})


@app.get("/api/generator/report")
def generator_report():
    gen = state.need_generator()
    return jsonable({"id": state.generator_id, "params": gen.model.num_params(),
                     "config": gen.model_config.to_dict(), "train_config": gen.train_config,
                     "metrics": gen.metrics, "history": gen.history})


# --- static frontend ------------------------------------------------------------------------
if paths.FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=paths.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        candidate = (paths.FRONTEND_DIST / full_path).resolve()
        if full_path and candidate.is_file() and paths.FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(paths.FRONTEND_DIST / "index.html")
