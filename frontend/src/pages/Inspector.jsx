import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api.js";
import { ActivationMap, AttentionMap } from "../components/Heatmap.jsx";
import {
  Badge, Button, DET, ErrorBox, GEN, GRID, MUTED, OK, Panel, ProbBar, Slider, Stat, Title, WARN, axisProps, fmt, tooltipProps,
} from "../components/ui.jsx";

export default function Inspector({ status, target }) {
  return (
    <div className="space-y-6">
      <section>
        <div className="label mb-1">Model Inspector</div>
        <h2 className="text-xl md:text-2xl font-semibold">Dentro de los modelos</h2>
        <p className="text-sm text-muted mt-1 max-w-3xl">
          Todo lo que ves aquí se lee directamente de los tensores y árboles reales: probabilidades de cada
          carácter, matrices de atención, activaciones internas y la contribución de cada característica a la decisión.
        </p>
      </section>
      <GeneratorInspector status={status} target={target} />
      <ClassifierInspector status={status} target={target} />
    </div>
  );
}

/* ------------------------------------------------------------------ generator */
function GeneratorInspector({ status, target }) {
  const [p, setP] = useState({ style: "phish", temperature: 0.8, top_k: 20, seed: 7, prefix: "", force_full: false });
  const [trace, setTrace] = useState(null);
  const [step, setStep] = useState(0);
  const [layer, setLayer] = useState(0);
  const [actLayer, setActLayer] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const timer = useRef(null);

  async function run(params = p) {
    setBusy(true);
    setError(null);
    try {
      const t = await api.traceGenerator(params);
      setTrace(t);
      setStep(0);
      setLayer(0);
      setActLayer(t.activations.length - 1);
      setPlaying(!params.force_full);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (target?.name) {
      const np = { ...p, style: target.style || p.style, prefix: target.name, force_full: true };
      setP(np);
      run(np);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target?.at]);

  useEffect(() => {
    if (!playing || !trace) return;
    timer.current = setInterval(() => {
      setStep((s) => {
        if (s >= trace.steps.length - 1) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 380);
    return () => clearInterval(timer.current);
  }, [playing, trace]);

  const cur = trace?.steps[step];
  const series = useMemo(
    () => trace?.steps.map((s, i) => ({ i, char: s.chosen === "<eos>" ? "⏹" : s.chosen, entropy: s.entropy_model, entropy_s: s.entropy_sample, p: s.p_model })) ?? [],
    [trace],
  );
  const norms = useMemo(() => {
    if (!trace) return [];
    return trace.tokens.map((tk, i) => {
      const row = { i, tk };
      trace.activations.forEach((L, li) => { row[trace.activation_labels[li]] = Math.sqrt(L[i].reduce((a, v) => a + v * v, 0)); });
      return row;
    });
  }, [trace]);

  const set = (k) => (v) => setP((o) => ({ ...o, [k]: v }));
  const nHeads = trace?.attention[0]?.length ?? 0;
  const shown = trace ? trace.tokens.slice(0, step + 1) : [];

  return (
    <Panel glow="gen" className="p-5">
      <Title sub="Generación carácter a carácter con las probabilidades reales del modelo" right={trace && !trace.settings.force_full && <Badge color={GEN}>generado en {fmt.ms(trace.generation_ms)}</Badge>}>
        <span style={{ color: GEN }}>🔴 Generador</span> — ¿cómo decide cada carácter?
      </Title>

      <div className="grid gap-4 md:grid-cols-[repeat(4,1fr)_1.6fr_auto] items-end">
        <div>
          <div className="text-xs text-muted mb-1">Estilo</div>
          <select value={p.style} onChange={(e) => set("style")(e.target.value)}>
            <option value="phish">phish</option>
            <option value="legit">legit</option>
          </select>
        </div>
        <Slider label="Temperatura" value={p.temperature} min={0} max={2} step={0.05} onChange={set("temperature")} accent={GEN} format={(v) => v.toFixed(2)} />
        <Slider label="Top-K" value={p.top_k} min={0} max={42} step={1} onChange={set("top_k")} accent={GEN} format={(v) => (v || "∞")} />
        <div>
          <div className="text-xs text-muted mb-1">Semilla</div>
          <input type="number" value={p.seed} onChange={(e) => set("seed")(Number(e.target.value) || 0)} />
        </div>
        <div>
          <div className="text-xs text-muted mb-1">Texto inicial forzado (opcional)</div>
          <input type="text" value={p.prefix} placeholder="p. ej. secure-" onChange={(e) => set("prefix")(e.target.value.toLowerCase())} />
          <label className="flex items-center gap-2 text-[11px] text-muted mt-1">
            <input type="checkbox" checked={p.force_full} onChange={(e) => set("force_full")(e.target.checked)} />
            puntuar este texto completo (no generar)
          </label>
        </div>
        <Button variant="gen" onClick={() => run()} disabled={busy || !status?.generator}>{p.force_full ? "Puntuar" : "Generar"}</Button>
      </div>
      <ErrorBox error={error} />

      {trace && (
        <div className="mt-5 space-y-5">
          <div className="rounded-xl border border-line bg-[#080b10] p-4">
            <div className="flex flex-wrap items-center gap-3 mb-3">
              <Button onClick={() => setPlaying((v) => !v)}>{playing ? "⏸ Pausa" : "▶ Reproducir"}</Button>
              <Button onClick={() => { setPlaying(false); setStep((s) => Math.max(0, s - 1)); }}>←</Button>
              <Button onClick={() => { setPlaying(false); setStep((s) => Math.min(trace.steps.length - 1, s + 1)); }}>→</Button>
              <span className="text-xs text-muted font-mono">paso {step + 1}/{trace.steps.length}</span>
              <span className="ml-auto text-xs font-mono text-muted">
                perplejidad {fmt.num(trace.perplexity, 2)} · log-prob {fmt.num(trace.logprob, 2)} · {trace.valid ? <span style={{ color: OK }}>sintácticamente válido</span> : <span style={{ color: WARN }}>inválido</span>}
              </span>
            </div>
            <div className="font-mono text-2xl md:text-3xl flex flex-wrap gap-y-2 select-text">
              {shown.map((tk, i) => {
                const att = trace.attention[layer];
                const w = att.reduce((a, h) => a + (h[step]?.[i] ?? 0), 0) / att.length;
                return (
                  <span key={i} onClick={() => { setPlaying(false); setStep(i); }}
                    className="relative px-[1px] cursor-pointer rounded"
                    style={{ background: `rgba(61,165,255,${w * 0.9})` }}
                    title={`atención media (capa ${layer + 1}) al elegir el carácter ${step + 1}: ${w.toFixed(3)}`}>
                    {tk.startsWith("<") ? <span className="text-xs align-middle px-1 rounded" style={{ color: GEN, border: `1px solid ${GEN}55` }}>{tk}</span> : tk}
                  </span>
                );
              })}
              <span className="px-1 rounded" style={{ color: GEN, borderBottom: `2px solid ${GEN}` }}>
                {cur.chosen === "<eos>" ? "⏹" : cur.chosen}
              </span>
            </div>
            <div className="text-[11px] text-muted mt-2">
              El fondo azul de cada carácter = atención media (capa {layer + 1}) que pone el modelo en él al elegir el siguiente.
              {cur.forced && " Este carácter fue forzado por el usuario."}
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <div className="label mb-2">candidatos para el paso {step + 1} · contexto «{cur.context || "∅"}»</div>
              <div className="space-y-1.5">
                {cur.candidates.map((c) => (
                  <div key={c.token} className="grid grid-cols-[48px_1fr_120px] gap-2 items-center text-sm">
                    <span className="font-mono" style={{ color: c.token === cur.chosen ? GEN : undefined }}>
                      {c.token === "<eos>" ? "⏹ fin" : `«${c.token}»`}
                    </span>
                    <div className="relative h-4 rounded bg-[#151c27] overflow-hidden">
                      <div className="absolute inset-y-0 left-0" style={{ width: `${c.p_model * 100}%`, background: `${MUTED}88` }} />
                      <div className="absolute inset-y-1 left-0" style={{ width: `${c.p_sample * 100}%`, background: c.token === cur.chosen ? GEN : DET }} />
                    </div>
                    <span className="font-mono text-[11px] text-right whitespace-nowrap">{fmt.pct(c.p_model)} → {fmt.pct(c.p_sample)}</span>
                  </div>
                ))}
              </div>
              <div className="text-[11px] text-muted mt-2">
                Gris = probabilidad del modelo (T=1). Color = distribución realmente muestreada tras temperatura {trace.settings.temperature} y top-K {trace.settings.top_k || "∞"}.
                Entropía: {fmt.num(cur.entropy_model, 2)} bits (modelo) → {fmt.num(cur.entropy_sample, 2)} bits (muestreo).
              </div>
            </div>
            <div>
              <div className="label mb-2">incertidumbre por paso (entropía, bits)</div>
              <ResponsiveContainer width="100%" height={210}>
                <LineChart data={series} margin={{ left: -20, right: 8, top: 6 }} onClick={(e) => e?.activeTooltipIndex != null && setStep(Number(e.activeTooltipIndex))}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="char" interval={0} {...axisProps} />
                  <YAxis {...axisProps} />
                  <Tooltip {...tooltipProps} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <ReferenceLine x={series[step]?.char} stroke="#fff" strokeOpacity={0.4} />
                  <Line dataKey="entropy" name="modelo" stroke={GEN} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                  <Line dataKey="entropy_s" name="muestreo" stroke={DET} strokeWidth={1.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
                  <Line dataKey="p" name="P(elegido)" stroke={OK} strokeWidth={1} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div>
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <div className="label">mapas de atención</div>
              {trace.attention.map((_, li) => (
                <button key={li} onClick={() => setLayer(li)} className={`text-xs px-2 py-0.5 rounded border cursor-pointer ${li === layer ? "border-det text-det" : "border-line text-muted"}`}>capa {li + 1}</button>
              ))}
              <span className="text-[11px] text-muted">filas = posición que decide (query) · columnas = posiciones a las que mira (key) · marco = paso actual</span>
            </div>
            <div className="flex flex-wrap gap-4">
              {Array.from({ length: nHeads }, (_, h) => (
                <AttentionMap key={h} title={`cabeza ${h + 1}`} matrix={trace.attention[layer][h]} tokens={trace.tokens} highlightRow={step} size={Math.min(240, 60 + trace.tokens.length * 9)} />
              ))}
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
            <div>
              <div className="flex flex-wrap items-center gap-2 mb-2">
                <div className="label">activaciones</div>
                {trace.activation_labels.map((l, li) => (
                  <button key={l} onClick={() => setActLayer(li)} className={`text-xs px-2 py-0.5 rounded border cursor-pointer ${li === actLayer ? "border-gen text-gen" : "border-line text-muted"}`}>{l}</button>
                ))}
              </div>
              <ActivationMap matrix={trace.activations[actLayer]} tokens={trace.tokens} highlightRow={step} width={Math.min(640, trace.activations[0][0].length * 8)} />
            </div>
            <div>
              <div className="label mb-2">norma L2 del flujo residual por posición</div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={norms} margin={{ left: -20, right: 8, top: 6 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="tk" interval={0} tickFormatter={(t) => (t.startsWith("<") ? "◆" : t)} {...axisProps} />
                  <YAxis {...axisProps} />
                  <Tooltip {...tooltipProps} formatter={(v) => fmt.num(v, 2)} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {trace.activation_labels.map((l, li) => (
                    <Line key={l} dataKey={l} stroke={[MUTED, DET, GEN, OK, WARN, "#c084fc", "#f472b6"][li % 7]} dot={false} strokeWidth={1.5} isAnimationActive={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------ classifier */
function ClassifierInspector({ status, target }) {
  const [domain, setDomain] = useState("secure-login-verify2931.web");
  const [res, setRes] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  async function run(d = domain) {
    setError(null);
    try {
      setRes(await api.inspectClassifier(d));
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    if (target?.name) {
      setDomain(target.name);
      run(target.name);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target?.at]);

  useEffect(() => {
    if (status?.classifier?.id) api.classifierReport().then(setReport).catch(() => {});
  }, [status?.classifier?.id]);

  const contrib = res?.contributions.slice(0, 12).map((c) => ({ ...c, label: c.feature })).reverse() ?? [];
  const votes = useMemo(() => {
    if (!res) return [];
    const b = Array.from({ length: 10 }, (_, i) => ({ bin: (i / 10).toFixed(1), n: 0, x: i / 10 }));
    res.tree_votes.forEach((v) => { b[Math.min(9, Math.floor(v * 10))].n += 1; });
    return b;
  }, [res]);
  const imp = report?.metrics.feature_importance.slice(0, 12).reverse() ?? [];

  return (
    <Panel glow="det" className="p-5">
      <Title sub="Qué características empujan la decisión y cuánto, árbol por árbol" right={res && <Badge color={DET}>{fmt.ms(res.ms)}</Badge>}>
        <span style={{ color: DET }}>🔵 Detector</span> — ¿por qué lo marca?
      </Title>
      <div className="flex gap-2">
        <input type="text" value={domain} onChange={(e) => setDomain(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()} placeholder="escribe cualquier dominio (no se contacta)" />
        <Button variant="det" onClick={() => run()} disabled={!status?.classifier}>Analizar</Button>
      </div>
      <ErrorBox error={error} />

      {res && (
        <div className="mt-5 grid gap-5 lg:grid-cols-[1fr_1.3fr]">
          <div className="space-y-4">
            <div className="rounded-xl border border-line bg-[#080b10] p-4">
              <div className="text-[11px] text-muted">nombre analizado (normalizado: sin www. ni TLD)</div>
              <div className="font-mono text-lg break-all">{res.name}</div>
              <div className="flex items-end justify-between mt-3">
                <div>
                  <div className="label">P(sospechoso)</div>
                  <div className="font-mono text-4xl" style={{ color: res.suspicious ? GEN : DET }}>{res.probability.toFixed(3)}</div>
                </div>
                {res.suspicious ? <Badge color={GEN} solid>SOSPECHOSO</Badge> : <Badge color={DET}>NO SOSPECHOSO</Badge>}
              </div>
              <div className="mt-2"><ProbBar p={res.probability} threshold={res.threshold} /></div>
              <div className="text-[11px] text-muted mt-2">
                Base (media del bosque sin mirar nada) {res.prior.toFixed(3)} + suma de contribuciones{" "}
                {(res.probability - res.prior >= 0 ? "+" : "") + (res.probability - res.prior).toFixed(3)} = {res.probability.toFixed(3)}. Umbral {res.threshold}.
              </div>
            </div>
            <div>
              <div className="label mb-1">votos de los {res.tree_votes.length} árboles</div>
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={votes} margin={{ left: -20, right: 4, top: 4 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="bin" {...axisProps} />
                  <YAxis allowDecimals={false} {...axisProps} />
                  <Tooltip {...tooltipProps} cursor={{ fill: "#ffffff08" }} />
                  <Bar dataKey="n" name="árboles" radius={[3, 3, 0, 0]}>
                    {votes.map((v) => <Cell key={v.bin} fill={v.x >= res.threshold ? GEN : DET} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="text-[11px] text-muted">Dispersión = desacuerdo entre árboles (confianza del bosque).</div>
            </div>
          </div>
          <div>
            <div className="label mb-1">contribución de cada característica a esta decisión</div>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={contrib} layout="vertical" margin={{ left: 40, right: 12 }}>
                <CartesianGrid stroke={GRID} horizontal={false} />
                <XAxis type="number" {...axisProps} tickFormatter={(v) => v.toFixed(2)} />
                <YAxis type="category" dataKey="label" width={130} interval={0} {...axisProps} />
                <Tooltip {...tooltipProps} cursor={{ fill: "#ffffff08" }} formatter={(v) => (v >= 0 ? "+" : "") + v.toFixed(4)}
                  labelFormatter={(l) => { const c = contrib.find((x) => x.label === l); return c ? `${c.description} = ${fmt.num(c.value, 3)}` : l; }} />
                <ReferenceLine x={0} stroke={MUTED} />
                <Bar dataKey="contribution" name="contribución">
                  {contrib.map((c) => <Cell key={c.feature} fill={c.contribution >= 0 ? GEN : DET} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="overflow-x-auto scroll-thin mt-2">
              <table className="w-full text-xs font-mono">
                <thead className="text-muted"><tr className="text-left">
                  <th className="font-normal py-1">característica</th><th className="font-normal">valor</th>
                  <th className="font-normal">mediana legít.</th><th className="font-normal">mediana phish.</th><th className="font-normal">contrib.</th>
                </tr></thead>
                <tbody>
                  {res.contributions.slice(0, 8).map((c) => (
                    <tr key={c.feature} className="border-t border-line" title={c.description}>
                      <td className="py-1">{c.feature}</td><td>{fmt.num(c.value, 2)}</td>
                      <td style={{ color: DET }}>{fmt.num(c.median.legitimate, 2)}</td><td style={{ color: GEN }}>{fmt.num(c.median.phishing, 2)}</td>
                      <td style={{ color: c.contribution >= 0 ? GEN : DET }}>{(c.contribution >= 0 ? "+" : "") + c.contribution.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {report && (
        <div className="mt-6 grid gap-5 lg:grid-cols-2">
          <div>
            <div className="label mb-1">importancia global de las características</div>
            <ResponsiveContainer width="100%" height={380}>
              <BarChart data={imp} layout="vertical" margin={{ left: 40, right: 12 }}>
                <CartesianGrid stroke={GRID} horizontal={false} />
                <XAxis type="number" {...axisProps} />
                <YAxis type="category" dataKey="feature" width={130} interval={0} {...axisProps} />
                <Tooltip {...tooltipProps} cursor={{ fill: "#ffffff08" }} formatter={(v) => v.toFixed(4)} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="impurity" name="impureza (Gini)" fill={DET} />
                <Bar dataKey="permutation" name="permutación (ΔF1 en val)" fill={GEN} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div>
            <div className="label mb-1">distribución de probabilidades en test por etiqueta real</div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={report.metrics.test_histogram} margin={{ left: -10, right: 8, top: 6 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="bin" {...axisProps} />
                <YAxis {...axisProps} />
                <Tooltip {...tooltipProps} cursor={{ fill: "#ffffff08" }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="legitimate" name="legítimos" fill={DET} />
                <Bar dataKey="phishing" name="phishing" fill={GEN} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </Panel>
  );
}
