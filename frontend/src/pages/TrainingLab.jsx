import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, subscribeJob } from "../api.js";
import {
  Badge, Button, DET, ErrorBox, Field, GEN, GRID, MUTED, OK, Panel, Slider, Stat, Title, WARN, axisProps, fmt, retrain, tooltipProps,
} from "../components/ui.jsx";

const VOCAB = 42;
// Exact parameter count of MiniGPT (tied output head), mirrors backend/phantomlink/generator/model.py
const paramCount = ({ n_layer, n_embd: E, block_size: B }) => VOCAB * E + B * E + n_layer * (12 * E * E + 13 * E) + 2 * E;

export default function TrainingLab({ status, refresh }) {
  const [presets, setPresets] = useState(null);
  const [job, setJob] = useState(null); // {id, kind, events, status}
  const [error, setError] = useState(null);
  const unsub = useRef(null);

  useEffect(() => { api.presets().then(setPresets).catch((e) => setError(e.message)); }, []);

  const attach = (id, kind) => {
    unsub.current?.();
    setJob({ id, kind, events: [], status: "running" });
    unsub.current = subscribeJob(id, (ev) => {
      setJob((j) => {
        if (!j || j.id !== id || j.events.some((e) => e.seq === ev.seq)) return j;
        return { ...j, events: [...j.events, ev], status: ev.type === "end" ? ev.status : j.status };
      });
      if (ev.type === "end") refresh();
    });
  };

  // Re-attach to a training that is already running (e.g. after a page reload).
  useEffect(() => {
    const j = status?.job;
    if (j && j.status === "running" && job?.id !== j.id) attach(j.id, j.kind);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status?.job?.id, status?.job?.status]);
  useEffect(() => () => unsub.current?.(), []);

  const running = job?.status === "running" || status?.job?.status === "running";

  async function start(kind, cfg) {
    setError(null);
    try {
      const r = kind === "generator" ? await api.trainGenerator(cfg) : await api.trainClassifier(cfg);
      attach(r.job_id, kind);
      refresh();
    } catch (e) {
      setError(e.message);
    }
  }
  async function stop() {
    try { await api.stop(); } catch (e) { setError(e.message); }
  }

  return (
    <div className="space-y-6">
      <section>
        <div className="label mb-1">Training Lab</div>
        <h2 className="text-xl md:text-2xl font-semibold">Entrena, rompe y compara los modelos</h2>
        <p className="text-sm text-muted mt-1 max-w-3xl">
          Cada gráfica se alimenta en directo (Server-Sent Events) del entrenamiento real. Solo puede haber un
          entrenamiento a la vez; al terminar, el modelo se guarda en <span className="font-mono">runs/</span> y pasa a ser el activo.
        </p>
      </section>
      <ErrorBox error={error} />
      {presets && (
        <>
          <GeneratorLab presets={presets} status={status} job={job?.kind === "generator" ? job : null} running={running} onStart={(c) => start("generator", c)} onStop={stop} />
          <ClassifierLab presets={presets} status={status} job={job?.kind === "classifier" ? job : null} running={running} onStart={(c) => start("classifier", c)} onStop={stop} />
        </>
      )}
      <Checkpoints status={status} running={running} refresh={refresh} />
    </div>
  );
}

function PresetBar({ presets, active, onPick, color }) {
  return (
    <div className="grid sm:grid-cols-3 gap-2 mb-4">
      {Object.entries(presets).map(([k, p]) => (
        <button key={k} onClick={() => onPick(k)} className={`text-left rounded-xl border p-3 cursor-pointer transition ${active === k ? "" : "border-line hover:border-[#2c3a4d]"}`}
          style={active === k ? { borderColor: color, background: `${color}12` } : {}}>
          <div className="text-sm font-semibold">{p.label}</div>
          <div className="text-[11px] text-muted mt-1 leading-snug">{p.expectation}</div>
        </button>
      ))}
    </div>
  );
}

function StageLog({ events }) {
  const stages = events.filter((e) => ["stage", "error", "done", "end"].includes(e.type));
  if (!stages.length) return null;
  return (
    <div className="font-mono text-[11px] text-muted space-y-0.5 max-h-28 overflow-y-auto scroll-thin rounded-lg border border-line bg-[#080b10] p-2">
      {stages.map((e) => (
        <div key={e.seq} style={{ color: e.type === "error" || e.stage === "diverged" ? GEN : e.stage === "saved" ? OK : undefined }}>
          {new Date(e.t * 1000).toLocaleTimeString()} · {e.message ?? (e.type === "end" ? `fin: ${e.status}` : e.type)}
        </div>
      ))}
    </div>
  );
}

function MiniChart({ data, x, lines, height = 170, y, title, log }) {
  return (
    <div>
      <div className="label mb-1">{title}</div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ left: -12, right: 8, top: 6 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey={x} type="number" domain={["dataMin", "dataMax"]} {...axisProps} />
          <YAxis domain={y ?? ["auto", "auto"]} scale={log ? "log" : "auto"} allowDataOverflow={!!log} {...axisProps} width={52}
            tickFormatter={(v) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : v !== 0 && Math.abs(v) < 0.01 ? v.toExponential(1) : Number(v.toPrecision(3)))} />
          <Tooltip {...tooltipProps} formatter={(v) => (typeof v === "number" ? Number(v.toPrecision(4)) : v)} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {lines.map((l) => (
            <Line key={l.key} dataKey={l.key} name={l.name} stroke={l.color} strokeWidth={l.w ?? 1.6} dot={l.dot ?? false} connectNulls isAnimationActive={false} strokeDasharray={l.dash} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ generator */
function GeneratorLab({ presets, status, job, running, onStart, onStop }) {
  const [preset, setPreset] = useState("balanced");
  const [cfg, setCfg] = useState({ ...presets.defaults.generator, ...presets.generator.balanced.config });
  const [report, setReport] = useState(null);
  const set = (k) => (v) => setCfg((c) => ({ ...c, [k]: v }));
  const pick = (k) => { setPreset(k); setCfg((c) => ({ ...c, ...presets.generator[k].config })); };

  useEffect(() => {
    if (status?.generator?.id) api.generatorReport().then(setReport).catch(() => {});
  }, [status?.generator?.id]);

  const live = job?.events.filter((e) => e.type === "metrics") ?? [];
  const history = live.length ? live : report?.history ?? [];
  const evals = history.filter((h) => h.val_loss != null);
  const last = history[history.length - 1];
  const lastSamples = [...history].reverse().find((h) => h.samples)?.samples;
  const params = paramCount(cfg);
  const headsOk = cfg.n_embd % cfg.n_head === 0;
  const isLive = live.length > 0;
  const modelInfo = job?.events.find((e) => e.stage === "model");

  return (
    <Panel glow="gen" className="p-5">
      <Title sub="Transformer decoder-only a nivel de carácter, entrenado desde cero" right={
        <div className="flex gap-2 items-center">
          {job?.status === "running" && <Badge color={WARN}>● entrenando</Badge>}
          {job && job.status !== "running" && <Badge color={job.status === "completed" ? OK : WARN}>{job.status}</Badge>}
        </div>}>
        <span style={{ color: GEN }}>🔴 Generador</span>
      </Title>
      <PresetBar presets={presets.generator} active={preset} onPick={pick} color={GEN} />

      <div className="grid gap-5 xl:grid-cols-[340px_1fr]">
        <div className="space-y-3">
          <div className="rounded-xl border border-warn/30 p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs">Arquitectura</span>{retrain}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Capas"><input type="number" min={1} max={6} value={cfg.n_layer} onChange={(e) => set("n_layer")(+e.target.value)} /></Field>
              <Field label="Cabezas de atención"><input type="number" min={1} max={8} value={cfg.n_head} onChange={(e) => set("n_head")(+e.target.value)} /></Field>
              <Field label="Dim. embeddings"><input type="number" min={16} max={256} step={8} value={cfg.n_embd} onChange={(e) => set("n_embd")(+e.target.value)} /></Field>
              <Field label="Contexto (tokens)"><input type="number" min={16} max={64} value={cfg.block_size} onChange={(e) => set("block_size")(+e.target.value)} /></Field>
            </div>
            <div className="text-[11px] text-muted mt-2">Cambiar la arquitectura crea otro modelo desde cero. La temperatura y el top-K, en cambio, se ajustan en tiempo real en la Arena.</div>
          </div>
          <Slider label="Dropout" value={cfg.dropout} min={0} max={0.5} step={0.05} onChange={set("dropout")} accent={GEN} format={(v) => v.toFixed(2)} />
          <Slider label="Learning rate (log₁₀)" value={Math.log10(cfg.learning_rate)} min={-5} max={0} step={0.1} onChange={(v) => set("learning_rate")(Number((10 ** v).toPrecision(2)))} accent={GEN} format={() => cfg.learning_rate} />
          <div className="grid grid-cols-2 gap-3">
            <Field label="Batch size"><input type="number" min={4} max={256} value={cfg.batch_size} onChange={(e) => set("batch_size")(+e.target.value)} /></Field>
            <Field label="Iteraciones"><input type="number" min={10} max={20000} step={100} value={cfg.iterations} onChange={(e) => set("iterations")(+e.target.value)} /></Field>
            <Field label="Grad clip (0 = off)"><input type="number" min={0} max={10} step={0.5} value={cfg.grad_clip} onChange={(e) => set("grad_clip")(+e.target.value)} /></Field>
            <Field label="Semilla"><input type="number" value={cfg.seed} onChange={(e) => set("seed")(+e.target.value)} /></Field>
            <Field label="Warmup (fracción)"><input type="number" min={0} max={0.5} step={0.01} value={cfg.warmup_frac} onChange={(e) => set("warmup_frac")(+e.target.value)} /></Field>
            <Field label="Evaluar cada"><input type="number" min={10} max={5000} step={10} value={cfg.eval_interval} onChange={(e) => set("eval_interval")(+e.target.value)} /></Field>
          </div>
          <div className="rounded-lg border border-line bg-[#080b10] px-3 py-2 text-xs font-mono flex justify-between">
            <span className="text-muted">parámetros</span>
            <span style={{ color: params > 2e6 ? GEN : undefined }}>{fmt.int(params)}</span>
          </div>
          {!headsOk && <div className="text-xs text-gen">La dimensión de embeddings debe ser divisible por el número de cabezas.</div>}
          <div className="flex gap-2">
            <Button variant="gen" className="flex-1" disabled={running || !headsOk} onClick={() => onStart(cfg)}>▶ Entrenar generador</Button>
            <Button variant="danger" disabled={job?.status !== "running"} onClick={onStop}>■ Detener</Button>
          </div>
          {job && <StageLog events={job.events} />}
        </div>

        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
            {isLive ? <Badge color={WARN}>entrenamiento en directo</Badge> : <Badge color={MUTED}>historial del modelo activo ({status?.generator?.id})</Badge>}
            {modelInfo && <span className="font-mono">{fmt.int(modelInfo.params)} parámetros · {modelInfo.device} · corpus {fmt.int(modelInfo.train_corpus.sequences)} train / {fmt.int(modelInfo.val_corpus.sequences)} val</span>}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-2">
            <Stat label="iteración" value={last ? `${fmt.int(last.iter)}` : "—"} />
            <Stat label="loss train" value={fmt.num(last?.train_loss, 3)} color={GEN} />
            <Stat label="loss val" value={fmt.num(evals.at(-1)?.val_loss, 3)} color={DET} />
            <Stat label="perplexity" value={fmt.num(evals.at(-1)?.val_ppl, 2)} />
            <Stat label="|∇|" value={fmt.num(last?.grad_norm, 2)} color={last?.grad_norm > 5 ? GEN : undefined} />
            <Stat label="tokens/s" value={fmt.int(last && Math.round(last.tokens_per_sec))} />
            <Stat label="memoria" value={last ? `${last.rss_mb.toFixed(0)} MB` : "—"} sub={last?.cuda_mb != null ? `CUDA ${last.cuda_mb.toFixed(0)} MB` : `${fmt.num(last?.ms_per_iter, 1)} ms/iter`} />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <MiniChart title="loss (entropía cruzada)" data={history} x="iter" lines={[
              { key: "train_loss", name: "train (batch)", color: `${GEN}99`, w: 1 },
              { key: "eval_train_loss", name: "train (eval)", color: GEN, w: 2, dot: { r: 2 } },
              { key: "val_loss", name: "validación", color: DET, w: 2, dot: { r: 2 } },
            ]} />
            <MiniChart title="perplejidad en validación" data={evals} x="iter" lines={[{ key: "val_ppl", name: "perplexity", color: OK, w: 2, dot: { r: 2 } }]} />
            <MiniChart title="norma del gradiente (antes del clipping)" data={history} x="iter" log lines={[{ key: "grad_norm", name: "|∇|", color: WARN }]} />
            <MiniChart title="learning rate" data={history} x="iter" lines={[{ key: "lr", name: "lr", color: "#c084fc" }]} />
            <MiniChart title="tokens por segundo" data={history} x="iter" lines={[{ key: "tokens_per_sec", name: "tok/s", color: DET }]} />
            <MiniChart title="tiempo por iteración (ms) y memoria (MB)" data={history} x="iter" lines={[
              { key: "ms_per_iter", name: "ms/iter", color: GEN }, { key: "rss_mb", name: "RSS MB", color: MUTED, dash: "4 3" },
            ]} />
          </div>
          {lastSamples && (
            <div className="rounded-xl border border-line bg-[#080b10] p-3">
              <div className="label mb-2">muestras generadas en la última evaluación (iteración {[...history].reverse().find((h) => h.samples)?.iter})</div>
              <div className="grid sm:grid-cols-2 gap-2 font-mono text-sm">
                {Object.entries(lastSamples).map(([s, names]) => (
                  <div key={s}>
                    <Badge color={s === "phish" ? GEN : DET}>{s}</Badge>
                    {names.map((n, i) => <div key={i} className="truncate select-text">{n || "∅"}<span className="text-muted">.test</span></div>)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ classifier */
function ClassifierLab({ presets, status, job, running, onStart, onStop }) {
  const [preset, setPreset] = useState("balanced");
  const [cfg, setCfg] = useState({ ...presets.defaults.classifier, ...presets.classifier.balanced.config });
  const [report, setReport] = useState(null);
  const [thr, setThr] = useState(null);
  const [thrMetrics, setThrMetrics] = useState(null);
  const set = (k) => (v) => setCfg((c) => ({ ...c, [k]: v }));
  const pick = (k) => { setPreset(k); setCfg((c) => ({ ...c, ...presets.classifier[k].config })); };
  const maxTrain = status?.dataset?.splits?.train?.rows ?? 140000;

  useEffect(() => {
    if (status?.classifier?.id) api.classifierReport().then((r) => { setReport(r); setThr(r.config.threshold); setThrMetrics(null); }).catch(() => {});
  }, [status?.classifier?.id]);

  useEffect(() => {
    if (thr == null || !report || thr === report.config.threshold) { setThrMetrics(null); return; }
    const t = setTimeout(() => api.classifierReport(thr).then((r) => setThrMetrics(r.metrics.test_at_threshold)).catch(() => {}), 200);
    return () => clearTimeout(t);
  }, [thr, report]);

  const live = job?.events.filter((e) => e.type === "metrics") ?? [];
  const history = live.length ? live : report?.history ?? [];
  const m = thrMetrics ?? report?.metrics?.test;
  const val = report?.metrics?.val;

  return (
    <Panel glow="det" className="p-5">
      <Title sub="RandomForest sobre 19 características léxicas · métricas en datos no vistos" right={
        <div className="flex gap-2 items-center">
          {job?.status === "running" && <Badge color={WARN}>● entrenando</Badge>}
          {job && job.status !== "running" && <Badge color={job.status === "completed" ? OK : WARN}>{job.status}</Badge>}
        </div>}>
        <span style={{ color: DET }}>🔵 Detector</span>
      </Title>
      <PresetBar presets={presets.classifier} active={preset} onPick={pick} color={DET} />
      <div className="grid gap-5 xl:grid-cols-[340px_1fr]">
        <div className="space-y-3">
          <Slider label="Número de árboles" value={cfg.n_estimators} min={1} max={500} step={1} onChange={set("n_estimators")} />
          <Field label="Profundidad máxima (vacío = sin límite)">
            <input type="number" min={1} max={40} value={cfg.max_depth ?? ""} onChange={(e) => set("max_depth")(e.target.value === "" ? null : +e.target.value)} />
          </Field>
          <Slider label="Muestras mínimas por hoja" value={cfg.min_samples_leaf} min={1} max={200} step={1} onChange={set("min_samples_leaf")} />
          <Field label="Pesos de clase">
            <select value={cfg.class_weight ?? "none"} onChange={(e) => set("class_weight")(e.target.value === "none" ? null : e.target.value)}>
              <option value="balanced">balanced (compensa el desequilibrio)</option>
              <option value="none">ninguno</option>
            </select>
          </Field>
          <Slider label="Umbral de decisión" value={cfg.threshold} min={0.05} max={0.95} step={0.01} onChange={set("threshold")} format={(v) => v.toFixed(2)} />
          <Slider label="Tamaño del conjunto de entrenamiento" value={cfg.train_size} min={1000} max={maxTrain} step={1000} onChange={set("train_size")} format={fmt.int} />
          <Field label="Semilla"><input type="number" value={cfg.seed} onChange={(e) => set("seed")(+e.target.value)} /></Field>
          <div className="flex gap-2">
            <Button variant="det" className="flex-1" disabled={running} onClick={() => onStart(cfg)}>▶ Entrenar detector</Button>
            <Button variant="danger" disabled={job?.status !== "running"} onClick={onStop}>■ Detener</Button>
          </div>
          {job && <StageLog events={job.events} />}
        </div>

        <div className="space-y-4">
          <div className="flex flex-wrap gap-2 text-xs">
            {live.length ? <Badge color={WARN}>entrenamiento en directo</Badge> : <Badge color={MUTED}>historial del modelo activo ({status?.classifier?.id})</Badge>}
          </div>
          <MiniChart title="métricas en validación a medida que se añaden árboles" data={history} x="trees" y={[0.5, 1]} lines={[
            { key: "val_f1", name: "F1 val", color: DET, w: 2 },
            { key: "val_precision", name: "precision val", color: OK },
            { key: "val_recall", name: "recall val", color: GEN },
            { key: "val_auc", name: "AUC val", color: "#c084fc" },
            { key: "train_accuracy", name: "accuracy train", color: MUTED, dash: "4 3" },
          ]} height={200} />

          {report && m && !live.length && (
            <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="label">test (no visto) · {fmt.int(m.n)} dominios</div>
                  {thrMetrics && <Badge color={WARN}>umbral {thr} (recalculado)</Badge>}
                </div>
                <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
                  <Stat label="accuracy" value={fmt.num(m.accuracy, 3)} />
                  <Stat label="precision" value={fmt.num(m.precision, 3)} />
                  <Stat label="recall" value={fmt.num(m.recall, 3)} />
                  <Stat label="F1" value={fmt.num(m.f1, 3)} color={DET} />
                  <Stat label="AUC" value={fmt.num(m.roc_auc, 3)} />
                </div>
                <Confusion c={m.confusion} />
                {val && <div className="text-[11px] text-muted mt-2 font-mono">validación: F1 {fmt.num(val.f1, 3)} · AUC {fmt.num(val.roc_auc, 3)} · entrenado con {fmt.int(report.metrics.train_rows)} dominios en {report.metrics.train_seconds.toFixed(1)} s · inferencia {report.metrics.inference_ms_per_1000.toFixed(1)} ms / 1.000</div>}
              </div>
              <div>
                <Slider label="Explorar el umbral (sin reentrenar)" value={thr ?? 0.5} min={0.05} max={0.95} step={0.05} onChange={setThr} format={(v) => v.toFixed(2)} />
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={report.metrics.test_threshold_sweep} margin={{ left: -12, right: 8, top: 10 }}>
                    <CartesianGrid stroke={GRID} vertical={false} />
                    <XAxis dataKey="threshold" type="number" domain={[0, 1]} {...axisProps} />
                    <YAxis domain={[0, 1]} {...axisProps} />
                    <Tooltip {...tooltipProps} formatter={(v) => v.toFixed(3)} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <ReferenceLine x={thr} stroke="#fff" strokeOpacity={0.5} />
                    <Line dataKey="precision" stroke={OK} dot={false} isAnimationActive={false} />
                    <Line dataKey="recall" stroke={GEN} dot={false} isAnimationActive={false} />
                    <Line dataKey="f1" stroke={DET} strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
                <div className="text-[11px] text-muted">Precision/recall/F1 en test para cada umbral: muestra el compromiso entre falsos positivos y phishing no detectado.</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Panel>
  );
}

function Confusion({ c }) {
  const max = Math.max(c.tn, c.fp, c.fn, c.tp);
  const cell = (v, label, good) => (
    <div className="rounded-lg p-2 text-center border border-line" style={{ background: `${good ? DET : GEN}${Math.round(20 + (v / max) * 60).toString(16)}` }}>
      <div className="font-mono text-lg">{fmt.int(v)}</div>
      <div className="text-[10px] text-muted">{label}</div>
    </div>
  );
  return (
    <div className="mt-3">
      <div className="label mb-1">matriz de confusión</div>
      <div className="grid grid-cols-[90px_1fr_1fr] gap-1.5 text-xs items-center">
        <span />
        <span className="text-center text-muted">predicho legítimo</span>
        <span className="text-center text-muted">predicho sospechoso</span>
        <span className="text-muted">real legítimo</span>
        {cell(c.tn, "verdaderos negativos", true)}
        {cell(c.fp, "falsos positivos", false)}
        <span className="text-muted">real phishing</span>
        {cell(c.fn, "phishing no detectado", false)}
        {cell(c.tp, "phishing detectado", true)}
      </div>
    </div>
  );
}

function Checkpoints({ status, running, refresh }) {
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => { api.models().then(setModels).catch(() => {}); }, [status?.generator?.id, status?.classifier?.id]);
  async function activate(kind, id) {
    setError(null);
    try { setModels(await api.activate(kind, id)); refresh(); } catch (e) { setError(e.message); }
  }
  if (!models) return null;
  return (
    <Panel className="p-5">
      <Title sub="Modelos de ejemplo incluidos y los que entrenas desde esta página">Checkpoints</Title>
      <ErrorBox error={error} />
      <div className="grid gap-4 md:grid-cols-2">
        {["generator", "classifier"].map((kind) => (
          <div key={kind}>
            <div className="label mb-2" style={{ color: kind === "generator" ? GEN : DET }}>{kind === "generator" ? "generador (.pt)" : "detector (.joblib)"}</div>
            <div className="space-y-1">
              {models[kind].map((m) => (
                <div key={m.id} className="flex items-center gap-2 text-xs font-mono rounded-lg border border-line px-2 py-1.5">
                  <span className="truncate flex-1">{m.id}</span>
                  <Badge color={m.origin === "shipped" ? OK : MUTED}>{m.origin === "shipped" ? "ejemplo" : "run"}</Badge>
                  <span className="text-muted">{m.size_kb} KB</span>
                  {m.active ? <Badge color={kind === "generator" ? GEN : DET} solid>activo</Badge>
                    : <Button className="!py-0.5 !px-2 text-xs" disabled={running} onClick={() => activate(kind, m.id)}>activar</Button>}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
