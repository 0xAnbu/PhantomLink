import { useState } from "react";
import { api } from "../api.js";
import { Badge, Button, DET, ErrorBox, GEN, OK, Panel, Stat, Title } from "../components/ui.jsx";

// Human vs AI: real held-out test domains with their dataset label. The visitor answers
// first; then the detector's decision and the true label are revealed.
export default function Challenge({ threshold, ready }) {
  const [items, setItems] = useState(null);
  const [idx, setIdx] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [error, setError] = useState(null);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 100000));

  async function start() {
    setError(null);
    try {
      const r = await api.challenge(10, seed);
      setItems(r.items);
      setIdx(0);
      setAnswers([]);
      setSeed((s) => s + 1);
    } catch (e) {
      setError(e.message);
    }
  }

  const cur = items?.[idx];
  const answered = answers.length > idx;
  const answer = (guess) => {
    const ai = cur.probability >= threshold;
    setAnswers((a) => [...a, { guess, ai, truth: cur.is_phishing }]);
  };
  const human = answers.filter((a) => a.guess === a.truth).length;
  const ai = answers.filter((a) => a.ai === a.truth).length;
  const done = items && answers.length === items.length;

  return (
    <Panel className="p-5">
      <Title
        sub="Dominios reales del conjunto de test (nunca vistos por el detector) con su etiqueta verdadera. El TLD se sustituye por .test."
        right={<Button onClick={start} disabled={!ready}>{items ? "Nueva partida" : "Jugar"}</Button>}
      >
        🧠 Tú contra la IA — ¿sospechoso o no?
      </Title>
      <ErrorBox error={error} />
      {!items && <p className="text-sm text-muted">10 dominios, mitad phishing y mitad legítimos. Responde antes que el detector.</p>}
      {items && (
        <div className="grid gap-4 md:grid-cols-[1.4fr_1fr] items-start">
          <div>
            {!done ? (
              <div className="rounded-xl border border-line bg-[#0a0e14] p-5 text-center">
                <div className="label mb-2">dominio {idx + 1} / {items.length}</div>
                <div className="font-mono text-xl md:text-2xl break-all select-text">{cur.display}</div>
                {!answered ? (
                  <div className="flex gap-3 justify-center mt-5">
                    <Button variant="gen" onClick={() => answer(true)}>Sospechoso</Button>
                    <Button variant="det" onClick={() => answer(false)}>No sospechoso</Button>
                  </div>
                ) : (
                  <div className="mt-4 space-y-2 text-sm">
                    <Reveal a={answers[idx]} p={cur.probability} />
                    <Button onClick={() => setIdx((i) => i + 1)}>Siguiente →</Button>
                  </div>
                )}
              </div>
            ) : (
              <div className="rounded-xl border border-line bg-[#0a0e14] p-5 text-center">
                <div className="text-lg">
                  {human > ai ? "¡Has ganado a la IA! 🏆" : human === ai ? "Empate 🤝" : "La IA gana esta vez 🤖"}
                </div>
                <div className="text-xs text-muted mt-1">Pulsa «Nueva partida» para jugar con otros dominios.</div>
              </div>
            )}
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="tus aciertos" value={`${human}/${answers.length}`} color={OK} />
            <Stat label="aciertos IA" value={`${ai}/${answers.length}`} color={DET} />
            <div className="col-span-2 flex flex-wrap gap-1">
              {answers.map((a, i) => (
                <span key={i} className="font-mono text-[11px] rounded px-1.5 py-0.5 border border-line" title={items[i].display}>
                  {i + 1}: {a.guess === a.truth ? "✓" : "✗"} / IA {a.ai === a.truth ? "✓" : "✗"}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}

function Reveal({ a, p }) {
  return (
    <div className="flex flex-wrap justify-center gap-2 items-center">
      <span>Tú: <b style={{ color: a.guess === a.truth ? OK : GEN }}>{a.guess ? "sospechoso" : "no sospechoso"}</b></span>
      <span>· IA: <b style={{ color: a.ai === a.truth ? OK : GEN }}>{a.ai ? "sospechoso" : "no sospechoso"}</b> <span className="font-mono text-muted">(p={p.toFixed(2)})</span></span>
      <span>· Etiqueta real: <Badge color={a.truth ? GEN : DET} solid>{a.truth ? "PHISHING" : "LEGÍTIMO"}</Badge></span>
    </div>
  );
}
