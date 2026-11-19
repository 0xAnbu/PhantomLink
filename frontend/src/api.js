// Thin client for the FastAPI backend. Every number on screen comes from these calls.
async function request(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    let msg = data?.detail ?? res.statusText;
    if (Array.isArray(msg)) msg = msg.map((d) => `${d.loc?.slice(1).join(".")}: ${d.msg}`).join(" · ");
    throw new Error(msg);
  }
  return data;
}

export const api = {
  status: () => request("GET", "/api/status"),
  presets: () => request("GET", "/api/presets"),
  models: () => request("GET", "/api/models"),
  activate: (kind, id) => request("POST", "/api/models/activate", { kind, id }),
  arena: (body) => request("POST", "/api/arena", body),
  challenge: (n, seed) => request("GET", `/api/challenge?n=${n}&seed=${seed}`),
  classify: (domains) => request("POST", "/api/classify", { domains }),
  traceGenerator: (body) => request("POST", "/api/inspect/generator", body),
  inspectClassifier: (domain) => request("POST", "/api/inspect/classifier", { domain }),
  classifierReport: (threshold) =>
    request("GET", `/api/classifier/report${threshold != null ? `?threshold=${threshold}` : ""}`),
  generatorReport: () => request("GET", "/api/generator/report"),
  trainGenerator: (cfg) => request("POST", "/api/train/generator", cfg),
  trainClassifier: (cfg) => request("POST", "/api/train/classifier", cfg),
  stop: () => request("POST", "/api/train/stop"),
  currentJob: () => request("GET", "/api/jobs/current"),
};

// Server-Sent Events subscription to a training job.
export function subscribeJob(jobId, onEvent, since = 0) {
  const es = new EventSource(`/api/jobs/${jobId}/events?since=${since}`);
  const types = ["stage", "metrics", "done", "error", "end"];
  types.forEach((t) =>
    es.addEventListener(t, (e) => {
      const ev = JSON.parse(e.data);
      onEvent(ev);
      if (t === "end") es.close();
    }),
  );
  es.onerror = () => {
    /* the browser reconnects automatically; the server replays from Last-Event-ID */
  };
  return () => es.close();
}
