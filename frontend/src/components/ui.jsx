// Small shared UI primitives.
export const GEN = "#ff3d5a";
export const DET = "#3da5ff";
export const OK = "#34e3a0";
export const WARN = "#ffb547";
export const MUTED = "#7d8a9c";
export const GRID = "#1d2633";

export const fmt = {
  pct: (v, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)} %`),
  num: (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d)),
  int: (v) => (v == null ? "—" : Number(v).toLocaleString("es-ES")),
  ms: (v) => (v == null ? "—" : v < 1 ? `${(v * 1000).toFixed(0)} µs` : `${v.toFixed(1)} ms`),
};

export function Panel({ className = "", children, glow }) {
  const g = glow === "gen" ? "glow-gen" : glow === "det" ? "glow-det" : "";
  return <div className={`panel ${g} ${className}`}>{children}</div>;
}

export function Title({ children, right, sub }) {
  return (
    <div className="flex items-start justify-between gap-3 mb-3">
      <div>
        <h3 className="text-sm font-semibold tracking-wide">{children}</h3>
        {sub && <p className="text-xs text-muted mt-0.5">{sub}</p>}
      </div>
      {right}
    </div>
  );
}

export function Badge({ color = MUTED, children, solid }) {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium whitespace-nowrap"
      style={solid ? { background: color, color: "#06080c" } : { background: `${color}1f`, color, border: `1px solid ${color}44` }}
    >
      {children}
    </span>
  );
}

export function Button({ children, onClick, disabled, variant = "ghost", className = "", title }) {
  const base = "rounded-lg px-3 py-2 text-sm font-medium transition disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer";
  const styles = {
    ghost: "border border-line bg-[#0a0e14] hover:border-[#2c3a4d] hover:bg-[#0f141c]",
    gen: "bg-gen text-[#06080c] hover:brightness-110",
    det: "bg-det text-[#06080c] hover:brightness-110",
    danger: "border border-gen/50 text-gen hover:bg-gen/10",
  };
  return (
    <button title={title} className={`${base} ${styles[variant]} ${className}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

export function Slider({ label, value, min, max, step, onChange, accent = DET, format = (v) => v, hint, tag }) {
  return (
    <label className="block" style={{ "--accent": accent }}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-muted flex items-center gap-1.5">
          {label} {tag}
        </span>
        <span className="font-mono text-sm" style={{ color: accent }}>{format(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <div className="text-[11px] text-muted mt-0.5">{hint}</div>}
    </label>
  );
}

export function Field({ label, children, hint, tag }) {
  return (
    <label className="block">
      <div className="text-xs text-muted mb-1 flex items-center gap-1.5">{label} {tag}</div>
      {children}
      {hint && <div className="text-[11px] text-muted mt-0.5">{hint}</div>}
    </label>
  );
}

export function Stat({ label, value, color, sub }) {
  return (
    <div className="rounded-xl border border-line bg-[#0a0e14] px-3 py-2.5 min-w-0">
      <div className="label truncate">{label}</div>
      <div className="font-mono text-lg sm:text-xl mt-0.5 truncate" style={{ color }}>{value}</div>
      {sub && <div className="text-[11px] text-muted">{sub}</div>}
    </div>
  );
}

export function ProbBar({ p, threshold = 0.5 }) {
  if (p == null) return <div className="h-2 rounded bg-line/50" />;
  const c = p >= threshold ? GEN : DET;
  return (
    <div className="relative h-2 rounded bg-[#151c27] overflow-hidden">
      <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${p * 100}%`, background: c, transition: "width .5s" }} />
      <div className="absolute inset-y-[-2px] w-px bg-white/70" style={{ left: `${threshold * 100}%` }} />
    </div>
  );
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return <div className="rounded-lg border border-gen/40 bg-gen/10 px-3 py-2 text-sm text-gen">{String(error)}</div>;
}

export const realtime = <Badge color={OK}>tiempo real</Badge>;
export const retrain = <Badge color={WARN}>requiere reentrenar</Badge>;

export const tooltipProps = {
  contentStyle: { background: "#0c1016", border: "1px solid #1d2633", borderRadius: 8, fontSize: 12 },
  labelStyle: { color: MUTED },
  itemStyle: { padding: 0 },
};
export const axisProps = { stroke: MUTED, tick: { fill: MUTED, fontSize: 11 }, tickLine: false, axisLine: { stroke: GRID } };
