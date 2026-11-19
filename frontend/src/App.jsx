import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { Badge, DET, GEN, OK, WARN, fmt } from "./components/ui.jsx";
import Arena from "./pages/Arena.jsx";
import Inspector from "./pages/Inspector.jsx";
import TrainingLab from "./pages/TrainingLab.jsx";

const TABS = [
  { id: "arena", label: "AI Arena", icon: "⚔" },
  { id: "inspector", label: "Model Inspector", icon: "◎" },
  { id: "lab", label: "Training Lab", icon: "⚗" },
];

export default function App() {
  const [tab, setTab] = useState(() => {
    try { return localStorage.getItem("pl-tab") || "arena"; } catch { return "arena"; }
  });
  const [status, setStatus] = useState(null);
  const [offline, setOffline] = useState(false);
  const [inspectTarget, setInspectTarget] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.status());
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [refresh]);

  const go = (id) => {
    setTab(id);
    try { localStorage.setItem("pl-tab", id); } catch { /* ignore */ }
  };

  const inspect = (target) => {
    setInspectTarget({ ...target, at: Date.now() });
    go("inspector");
  };

  const job = status?.job;
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-line bg-bg/85 backdrop-blur">
        <div className="mx-auto max-w-[1400px] px-4 h-14 flex items-center gap-4">
          <div className="flex items-center gap-2.5 shrink-0">
            <span className="relative w-7 h-5">
              <span className="absolute left-0 top-0 w-5 h-5 rounded-full" style={{ background: GEN }} />
              <span className="absolute right-0 top-0 w-5 h-5 rounded-full mix-blend-screen" style={{ background: DET }} />
            </span>
            <span className="font-mono font-bold tracking-[.25em] text-sm">PHANTOMLINK</span>
            <span className="hidden sm:inline text-[11px] text-muted font-mono">AI vs AI</span>
          </div>
          <nav className="flex gap-1 overflow-x-auto scroll-thin">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => go(t.id)}
                className={`px-3 py-1.5 rounded-lg text-sm whitespace-nowrap cursor-pointer transition ${
                  tab === t.id ? "bg-[#151c27] text-text border border-line" : "text-muted hover:text-text"
                }`}
              >
                <span className="mr-1.5 opacity-70">{t.icon}</span>
                {t.label}
              </button>
            ))}
          </nav>
          <div className="ml-auto hidden md:flex items-center gap-2 text-[11px] font-mono text-muted">
            {offline ? (
              <Badge color={GEN}>backend desconectado</Badge>
            ) : (
              <>
                {job?.status === "running" && <Badge color={WARN}>● entrenando {job.kind}</Badge>}
                <Badge color={GEN}>GEN {status?.generator ? `${fmt.int(status.generator.params)} params` : "—"}</Badge>
                <Badge color={DET}>DET {status?.classifier ? `${status.classifier.config.n_estimators} árboles` : "—"}</Badge>
                <Badge color={OK}>{status?.device?.cuda_available ? "CUDA" : "CPU"}</Badge>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-4 py-6">
        {offline && (
          <div className="panel p-6 text-sm text-muted mb-6">
            No se puede contactar con el backend. Arráncalo con{" "}
            <code className="font-mono text-text">./run.sh</code> (o{" "}
            <code className="font-mono text-text">uvicorn phantomlink.api:app --app-dir backend</code>).
          </div>
        )}
        <div hidden={tab !== "arena"}><Arena status={status} onInspect={inspect} /></div>
        <div hidden={tab !== "inspector"}><Inspector status={status} target={inspectTarget} /></div>
        <div hidden={tab !== "lab"}><TrainingLab status={status} refresh={refresh} /></div>
      </main>

      <footer className="mx-auto max-w-[1400px] px-4 pb-8 text-[11px] text-muted leading-relaxed">
        Los dominios se muestran como texto con el TLD reservado <span className="font-mono">.test</span> (RFC 2606) y
        nunca se contactan. El detector solo analiza la forma del nombre: «no sospechoso» no significa seguro.
        Datos: PhiUSIIL Phishing URL Dataset (UCI, CC BY 4.0).
      </footer>
    </div>
  );
}
