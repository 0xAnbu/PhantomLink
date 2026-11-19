import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api.js";
import {
  Badge, Button, DET, ErrorBox, GEN, GRID, MUTED, OK, Panel, ProbBar, Slider, Stat, Title, WARN,
  axisProps, fmt, realtime, tooltipProps,
} from "../components/ui.jsx";
import Challenge from "./Challenge.jsx";

const STYLE_LABEL = { phish: "estilo phish", legit: "estilo legit" };

export default function Arena({ status, onInspect }) {
  const [gen, setGen] = useState({ temperature: 0.8, top_k: 20, max_len: 40, seed: 42, count: 20, mix: "mixed" });
  const [threshold, setThreshold] = useState(0.5);
  const [round, setRound] = useState(null);
  const [rounds, setRounds] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [autoSeed, setAutoSeed] = useState(true);

  const set = (k) => (v) => setGen((g) => ({ ...g, [k]: v }));

  async function duel() {
    setBusy(true);
    setError(null);
    try {
      const seed = autoSeed && round ? gen.seed + 1 : gen.seed;
      if (seed !== gen.seed) setGen((g) => ({ ...g, seed }));
      const r = await api.arena({ ...gen, seed, threshold });
      r.key = Date.now();
      setRound(r);
      setRounds((rs) => [...rs, summarize(r, rs.length + 1)].slice(-12));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  // The decision rule is the detector's (probability ≥ threshold). Moving the threshold
  // re-applies it to the probabilities the model already returned for this round.
  const items = useMemo(
    () => round?.items.map((it) => ({ ...it, suspicious: it.valid ? it.probability >= threshold : null })) ?? [],
    [round, threshold],
  );
  const valid = items.filter((i) => i.valid);
  const flagged = valid.filter((i) => i.suspicious).length;

  const hist = useMemo(() => {
    const bins = Array.from({ length: 10 }, (_, i) => ({ bin: `${i / 10}`, phish: 0, legit: 0, x: i / 10 }));
    valid.forEach((i) => { bins[Math.min(9, Math.floor(i.probability * 10))][i.style] += 1; });
    return bins;
  }, [valid]);

  const genInfo = status?.generator;
  const detInfo = status?.classifier;

  return (
    <div className="space-y-6">
      <section className="text-center pt-2 pb-1">
        <div className="label mb-2">AI Arena</div>
        <h1 className="text-2xl md:text-4xl font-semibold tracking-tight">
          Una IA <span style={{ color: GEN }}>inventa dominios</span>. Otra intenta{" "}
          <span style={{ color: DET }}>detectarlos</span>.
        </h1>
        <p className="text-muted text-sm mt-2 max-w-2xl mx-auto">
          Dos modelos entrenados desde cero: un Transformer que escribe nombres letra a letra y un
          Random Forest que juzga si su estructura se parece a la del phishing.
        </p>
      </section>

      <section className="grid gap-4 lg:grid-cols-[1fr_auto_1fr] items-stretch">
        <Panel glow="gen" className="p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="label" style={{ color: GEN }}>🔴 IA generadora</div>
              <div className="font-mono text-xs text-muted mt-1">
                MiniGPT · {genInfo ? `${genInfo.config.n_layer} capas · ${genInfo.config.n_head} cabezas · ${fmt.int(genInfo.params)} parámetros` : "—"}
              </div>
            </div>
            {genInfo?.metrics?.final_val_ppl && <Badge color={GEN}>ppl val {fmt.num(genInfo.metrics.final_val_ppl, 2)}</Badge>}
          </div>
          <div className="grid sm:grid-cols-2 gap-4">
            <Slider label="Temperatura (creatividad)" tag={realtime} value={gen.temperature} min={0} max={2} step={0.05} onChange={set("temperature")} accent={GEN} format={(v) => v.toFixed(2)} />
            <Slider label="Top-K" tag={realtime} value={gen.top_k} min={0} max={42} step={1} onChange={set("top_k")} accent={GEN} format={(v) => (v === 0 ? "sin límite" : v)} />
            <Slider label="Longitud máxima" value={gen.max_len} min={4} max={63} step={1} onChange={set("max_len")} accent={GEN} />
            <Slider label="Dominios por ronda" value={gen.count} min={4} max={100} step={2} onChange={set("count")} accent={GEN} />
          </div>
          <div className="grid grid-cols-[1fr_auto] gap-3 mt-4 items-end">
            <div>
              <div className="text-xs text-muted mb-1">Estilo solicitado</div>
              <div className="flex gap-1">
                {[["mixed", "mitad y mitad"], ["phish", "phish"], ["legit", "legit"]].map(([v, l]) => (
                  <button key={v} onClick={() => set("mix")(v)}
                    className={`flex-1 rounded-lg px-2 py-1.5 text-xs cursor-pointer border ${gen.mix === v ? "border-gen/60 text-gen bg-gen/10" : "border-line text-muted"}`}>
                    {l}
                  </button>
                ))}
              </div>
            </div>
            <div className="w-28">
              <div className="text-xs text-muted mb-1">Semilla</div>
              <input type="number" value={gen.seed} onChange={(e) => set("seed")(Number(e.target.value) || 0)} />
            </div>
          </div>
          <label className="flex items-center gap-2 text-[11px] text-muted mt-2">
            <input type="checkbox" checked={autoSeed} onChange={(e) => setAutoSeed(e.target.checked)} />
            incrementar la semilla en cada ronda (desmárcalo para repetir exactamente la misma ronda)
          </label>
        </Panel>

        <div className="flex lg:flex-col items-center justify-center gap-3 py-2">
          <div className="hidden lg:block font-mono text-muted text-xs">VS</div>
          <button
            onClick={duel}
            disabled={busy || !genInfo || !detInfo}
            className="relative rounded-2xl px-6 py-5 font-semibold text-[#06080c] cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed transition hover:scale-[1.03]"
            style={{ background: `linear-gradient(135deg, ${GEN}, ${DET})` }}
          >
            <span className={busy ? "" : "pulse absolute inset-0 rounded-2xl"} />
            <div className="text-lg leading-tight">⚡ INICIAR DUELO</div>
            <div className="text-xs font-mono opacity-80">{gen.count} DOMINIOS</div>
          </button>
          {round && <div className="font-mono text-[11px] text-muted">ronda #{rounds.length}</div>}
        </div>

        <Panel glow="det" className="p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="label" style={{ color: DET }}>🔵 IA detectora</div>
              <div className="font-mono text-xs text-muted mt-1">
                Random Forest · {detInfo ? `${detInfo.config.n_estimators} árboles · prof. ${detInfo.config.max_depth ?? "∞"} · 19 características` : "—"}
              </div>
            </div>
            {detInfo?.metrics?.test && <Badge color={DET}>F1 test {fmt.num(detInfo.metrics.test.f1, 3)}</Badge>}
          </div>
          <Slider label="Umbral de decisión (sensibilidad)" tag={realtime} value={threshold} min={0.05} max={0.95} step={0.01} onChange={setThreshold} accent={DET} format={(v) => v.toFixed(2)}
            hint="Probabilidad a partir de la cual el detector marca un dominio como sospechoso. Más bajo = más sensible." />
          {detInfo?.metrics?.test && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-4">
              <Stat label="accuracy" value={fmt.num(detInfo.metrics.test.accuracy, 3)} />
              <Stat label="precision" value={fmt.num(detInfo.metrics.test.precision, 3)} />
              <Stat label="recall" value={fmt.num(detInfo.metrics.test.recall, 3)} />
              <Stat label="AUC" value={fmt.num(detInfo.metrics.test.roc_auc, 3)} />
            </div>
          )}
          <p className="text-[11px] text-muted mt-3">
            Métricas medidas en {fmt.int(detInfo?.metrics?.test?.n)} dominios reales de test (no vistos en el entrenamiento),
            al umbral con el que se entrenó ({detInfo?.config?.threshold}).
          </p>
        </Panel>
      </section>

      <ErrorBox error={error} />

      {round && (
        <section className="grid gap-4 xl:grid-cols-[1.35fr_1fr]">
          <Panel className="p-4">
            <Title sub="Haz clic en un dominio para abrirlo en el Inspector." right={<span className="font-mono text-[11px] text-muted">gen {fmt.ms(round.summary.generation_ms)} · det {fmt.ms(round.summary.classification_ms)}</span>}>
              Dominios generados → decisión del detector
            </Title>
            <div className="hidden sm:grid grid-cols-[minmax(0,1fr)_64px_150px_150px] gap-x-3 mb-1">
              <div className="label">dominio generado (texto, no enlace)</div>
              <div className="label">estilo</div>
              <div className="label">P(sospechoso)</div>
              <div className="label text-right">decisión</div>
            </div>
            <div className="space-y-1.5 text-sm" key={round.key}>
              {items.map((it, i) => (
                <Row key={it.id} it={it} i={i} threshold={threshold} onInspect={onInspect} />
              ))}
            </div>
          </Panel>

          <div className="space-y-4">
            <Panel className="p-4">
              <Title sub="Resumen de esta ronda">Resultados del experimento</Title>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                <Stat label="generados" value={round.summary.generated} />
                <Stat label="válidos" value={valid.length} color={valid.length < items.length ? WARN : undefined} sub={`${round.summary.invalid} inválidos`} />
                <Stat label="únicos" value={round.summary.unique} sub={`${round.summary.in_dataset} ya en dataset`} />
                <Stat label="sospechosos" value={flagged} color={GEN} />
                <Stat label="no sospechosos" value={valid.length - flagged} color={DET} />
                <Stat label="P media" value={fmt.num(round.summary.mean_probability, 2)} />
              </div>
              <div className="mt-4 space-y-2">
                {Object.entries(round.summary.by_style).map(([s, v]) => {
                  const sub = valid.filter((i) => i.style === s);
                  const f = sub.filter((i) => i.suspicious).length;
                  return (
                    <div key={s}>
                      <div className="flex justify-between text-xs mb-1">
                        <span>{STYLE_LABEL[s]} <span className="text-muted">({sub.length} válidos)</span></span>
                        <span className="font-mono">{f}/{sub.length} marcados · {fmt.pct(sub.length ? f / sub.length : null, 0)}</span>
                      </div>
                      <div className="h-2 rounded bg-[#151c27] overflow-hidden">
                        <div className="h-full" style={{ width: `${sub.length ? (100 * f) / sub.length : 0}%`, background: s === "phish" ? GEN : DET }} />
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="text-[11px] text-muted mt-3">
                El estilo es lo que se pidió al generador, no una verdad comprobada: un «estilo phish» marcado es
                acuerdo entre dos modelos, no un phishing verificado.
              </p>
            </Panel>

            <Panel className="p-4">
              <Title sub="Probabilidades devueltas por el detector en esta ronda">Distribución de confianza</Title>
              <ResponsiveContainer width="100%" height={170}>
                <BarChart data={hist} margin={{ left: -20, right: 4, top: 4 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="bin" {...axisProps} />
                  <YAxis allowDecimals={false} {...axisProps} />
                  <Tooltip {...tooltipProps} cursor={{ fill: "#ffffff08" }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="phish" name="estilo phish" stackId="a" fill={GEN} />
                  <Bar dataKey="legit" name="estilo legit" stackId="a" fill={DET} radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Panel>
          </div>
        </section>
      )}

      {rounds.length > 0 && <RoundHistory rounds={rounds} />}

      {!round && (
        <Panel className="p-8 text-center text-muted text-sm">
          Ajusta la creatividad del generador o la sensibilidad del detector y pulsa{" "}
          <span className="text-text">⚡ Iniciar duelo</span>.
        </Panel>
      )}

      <Challenge threshold={threshold} ready={!!detInfo} />
    </div>
  );
}

function Row({ it, i, threshold, onInspect }) {
  const delay = { animationDelay: `${Math.min(i, 40) * 45}ms` };
  return (
    <div className="rowin grid grid-cols-[minmax(0,1fr)_auto] sm:grid-cols-[minmax(0,1fr)_64px_150px_150px] gap-x-3 gap-y-1 items-center border-b border-line/50 sm:border-0 pb-1.5 sm:pb-0" style={delay}>
      <button
        className="col-span-2 sm:col-span-1 text-left font-mono truncate cursor-pointer hover:text-white select-text min-w-0"
        title={it.valid ? "Inspeccionar" : it.invalid_reason}
        onClick={() => onInspect({ name: it.name, style: it.style })}
      >
        <span className={it.valid ? "" : "line-through text-muted"}>{it.display || "∅"}</span>
        {it.in_dataset && <span className="ml-2 text-[10px] text-warn" title={`El nombre existe en el split ${it.in_dataset} del dataset`}>● memorizado</span>}
        {it.duplicate && <span className="ml-2 text-[10px] text-muted">duplicado</span>}
      </button>
      <span className="hidden sm:block"><Badge color={it.style === "phish" ? GEN : DET}>{it.style}</Badge></span>
      <span className="flex items-center gap-2 min-w-0">
        <span className="sm:hidden"><Badge color={it.style === "phish" ? GEN : DET}>{it.style}</Badge></span>
        <span className="flex-1 min-w-[60px]"><ProbBar p={it.probability} threshold={threshold} /></span>
        <span className="font-mono text-[11px] w-8 text-right">{it.probability == null ? "" : it.probability.toFixed(2)}</span>
      </span>
      <span className="text-right">
        {!it.valid ? (
          <Badge color={MUTED}>inválido: {it.invalid_reason}</Badge>
        ) : it.suspicious ? (
          <Badge color={GEN} solid>SOSPECHOSO</Badge>
        ) : (
          <Badge color={DET}>NO SOSPECHOSO</Badge>
        )}
      </span>
    </div>
  );
}

function summarize(r, n) {
  const v = r.items.filter((i) => i.valid);
  const by = (s) => {
    const sub = v.filter((i) => i.style === s);
    return sub.length ? sub.filter((i) => i.probability >= r.settings.threshold).length / sub.length : null;
  };
  return {
    round: n, ...r.settings,
    valid: v.length / r.items.length,
    suspicious: v.length ? v.filter((i) => i.suspicious).length / v.length : null,
    phish: by("phish"), legit: by("legit"),
    mean: r.summary.mean_probability,
    novel: 1 - r.summary.in_dataset / r.items.length,
    models: r.models,
  };
}

function RoundHistory({ rounds }) {
  return (
    <Panel className="p-4">
      <Title sub="Compara cómo cambian las decisiones al modificar los parámetros">Historial de rondas</Title>
      <div className="grid gap-4 lg:grid-cols-[1fr_1.2fr]">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={rounds} margin={{ left: -18, right: 8, top: 6 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="round" {...axisProps} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} {...axisProps} />
            <Tooltip {...tooltipProps} formatter={(v) => fmt.pct(v, 0)} labelFormatter={(l) => `ronda ${l}`} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line dataKey="phish" name="phish marcados" stroke={GEN} strokeWidth={2} dot connectNulls />
            <Line dataKey="legit" name="legit marcados" stroke={DET} strokeWidth={2} dot connectNulls />
            <Line dataKey="valid" name="válidos" stroke={OK} strokeDasharray="4 3" dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full text-xs font-mono">
            <thead className="text-muted">
              <tr className="text-left">
                {["#", "temp", "top-k", "umbral", "semilla", "válidos", "nuevos", "phish→sosp.", "legit→sosp."].map((h) => (
                  <th key={h} className="py-1 pr-3 font-normal">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...rounds].reverse().map((r) => (
                <tr key={r.round} className="border-t border-line">
                  <td className="py-1 pr-3">{r.round}</td>
                  <td className="pr-3">{r.temperature.toFixed(2)}</td>
                  <td className="pr-3">{r.top_k || "∞"}</td>
                  <td className="pr-3">{r.threshold.toFixed(2)}</td>
                  <td className="pr-3">{r.seed}</td>
                  <td className="pr-3">{fmt.pct(r.valid, 0)}</td>
                  <td className="pr-3">{fmt.pct(r.novel, 0)}</td>
                  <td className="pr-3" style={{ color: GEN }}>{fmt.pct(r.phish, 0)}</td>
                  <td className="pr-3" style={{ color: DET }}>{fmt.pct(r.legit, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Panel>
  );
}
