"""Record social-media cuts of the demo (16:9 and 9:16) with captions, zooms and a cursor.

Dev-only tool. Requirements: `pip install playwright`, a local Chromium and a full ffmpeg build
(set FFMPEG=/path/to/ffmpeg if it is not on PATH). The app must be running (./run.sh).

    python scripts/record_social.py            # -> docs/demo_16x9.mp4, docs/demo_9x16.mp4
    python scripts/record_social.py --only 9x16

Realism rule: the app and the models run live. The script only clicks, scrolls, zooms the
camera and overlays captions. Every number in a caption is read from the API responses of this
same recording, never typed by hand.
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = os.environ.get("PHANTOMLINK_URL", "http://127.0.0.1:8000/")
DOCS = Path(__file__).resolve().parents[1] / "docs"
FORMATS = {
    "16x9": {"viewport": {"width": 1280, "height": 720}, "dsf": 1.5, "size": {"width": 1920, "height": 1080}},
    "9x16": {"viewport": {"width": 432, "height": 768}, "dsf": 2.5, "size": {"width": 1080, "height": 1920}},
}

# ----------------------------------------------------------------------------- in-page FX
FX_JS = r"""
(() => {
  if (window.__fx) return;
  const vertical = innerHeight > innerWidth;
  const css = document.createElement('style');
  css.textContent = `
    html { scrollbar-width: none; } ::-webkit-scrollbar { display: none; }
    #root { transition: transform .9s cubic-bezier(.2,.8,.2,1); will-change: transform; }
    .fx-layer { position: fixed; inset: 0; pointer-events: none; z-index: 99999; font-family: Inter, "DejaVu Sans", system-ui, sans-serif; }
    .fx-cap { position: absolute; left: 50%; transform: translate(-50%, 18px); opacity: 0;
      bottom: ${vertical ? '15%' : '7%'}; ${vertical ? 'width: 88%; box-sizing: border-box;' : 'max-width: 78%;'} text-align: center;
      background: rgba(6,8,12,.86); border: 1px solid rgba(255,255,255,.12); border-radius: 16px;
      padding: ${vertical ? '12px 16px' : '14px 26px'}; color: #fff; font-weight: 800; letter-spacing: -.01em;
      font-size: ${vertical ? '22px' : '30px'}; line-height: 1.22; box-shadow: 0 20px 60px -10px rgba(0,0,0,.8);
      backdrop-filter: blur(8px); transition: opacity .35s ease, transform .45s cubic-bezier(.2,.9,.3,1.3); }
    .fx-cap.on { opacity: 1; transform: translate(-50%, 0); }
    .fx-cap small { display: block; font-weight: 500; font-size: .58em; color: #9fb0c4; margin-top: 4px; letter-spacing: 0; }
    .fx-r { color: #ff3d5a; } .fx-b { color: #3da5ff; } .fx-g { color: #34e3a0; }
    .fx-card { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: ${vertical ? '18px' : '22px'}; text-align: center; padding: 8%; color: #fff; opacity: 0; transition: opacity .5s ease;
      background: radial-gradient(900px 500px at 25% 30%, #ff3d5a2e, transparent 60%), radial-gradient(900px 500px at 75% 70%, #3da5ff2e, transparent 60%), #06080c; }
    .fx-card.on { opacity: 1; }
    .fx-card h1 { margin: 0; font-size: ${vertical ? '38px' : '58px'}; line-height: 1.05; font-weight: 900; letter-spacing: -.03em; }
    .fx-card p { margin: 0; font-size: ${vertical ? '19px' : '26px'}; color: #c9d4e3; font-weight: 600; }
    .fx-card .fx-mono { font-family: "DejaVu Sans Mono", monospace; font-size: ${vertical ? '13px' : '16px'}; color: #7d8a9c; font-weight: 400; letter-spacing: .2em; }
    .fx-card .fx-stat { display: flex; gap: ${vertical ? '12px' : '28px'}; flex-wrap: wrap; justify-content: center; }
    .fx-card .fx-stat div { border: 1px solid #1d2633; background: #0c1016; border-radius: 16px; padding: 14px 20px; }
    .fx-card .fx-stat b { display: block; font-size: ${vertical ? '28px' : '40px'}; font-family: "DejaVu Sans Mono", monospace; }
    .fx-card .fx-stat span { font-size: ${vertical ? '13px' : '15px'}; color: #9fb0c4; }
    .fx-card > * { transform: translateY(14px); transition: transform .6s cubic-bezier(.2,.9,.3,1.2); }
    .fx-card.on > * { transform: none; }
    .fx-spot { position: absolute; border-radius: 16px; border: 2px solid #fff; opacity: 0;
      box-shadow: 0 0 0 9999px rgba(0,0,0,.45), 0 0 30px 4px var(--c, #fff); transition: all .45s ease; }
    .fx-spot.on { opacity: 1; }
    .fx-cursor { position: absolute; width: 22px; height: 22px; margin: -11px 0 0 -11px; border-radius: 50%;
      background: rgba(255,255,255,.95); box-shadow: 0 0 0 4px rgba(255,255,255,.25), 0 4px 14px rgba(0,0,0,.6);
      transition: left .65s cubic-bezier(.3,.7,.2,1), top .65s cubic-bezier(.3,.7,.2,1), transform .12s; opacity: 0; }
    .fx-cursor.on { opacity: 1; } .fx-cursor.down { transform: scale(.7); }
    .fx-ripple { position: absolute; width: 16px; height: 16px; margin: -8px 0 0 -8px; border-radius: 50%; border: 3px solid #fff;
      animation: fxr .6s ease-out forwards; }
    @keyframes fxr { to { transform: scale(4.5); opacity: 0; } }
  `;
  document.head.appendChild(css);
  const layer = document.createElement('div'); layer.className = 'fx-layer'; document.body.appendChild(layer);
  const cap = document.createElement('div'); cap.className = 'fx-cap'; layer.appendChild(cap);
  const card = document.createElement('div'); card.className = 'fx-card'; layer.appendChild(card);
  const spot = document.createElement('div'); spot.className = 'fx-spot'; layer.appendChild(spot);
  const cur = document.createElement('div'); cur.className = 'fx-cursor'; layer.appendChild(cur);
  cur.style.left = (innerWidth * .5) + 'px'; cur.style.top = (innerHeight * .8) + 'px';
  const root = document.getElementById('root');
  window.__fx = {
    vertical,
    caption(html) { if (!html) { cap.classList.remove('on'); return; }
      cap.classList.remove('on'); setTimeout(() => { cap.innerHTML = html; cap.classList.add('on'); }, cap.innerHTML ? 180 : 0); },
    card(html) { if (!html) { card.classList.remove('on'); return; } card.innerHTML = html; requestAnimationFrame(() => card.classList.add('on')); },
    spot(sel, color, pad = 8) { const el = typeof sel === 'string' ? document.querySelector(sel) : sel;
      if (!el) { spot.classList.remove('on'); return; } const r = el.getBoundingClientRect();
      Object.assign(spot.style, { left: (r.left - pad) + 'px', top: (r.top - pad) + 'px', width: (r.width + 2 * pad) + 'px', height: (r.height + 2 * pad) + 'px' });
      spot.style.setProperty('--c', color || '#fff'); spot.classList.add('on'); },
    unspot() { spot.classList.remove('on'); },
    zoom(el, s) { const r = el.getBoundingClientRect();
      root.style.transformOrigin = `${r.left + r.width / 2 + scrollX}px ${r.top + r.height / 2 + scrollY}px`;
      root.style.transform = `scale(${s})`; },
    unzoom() { root.style.transform = 'none'; },
    cursorTo(x, y) { cur.classList.add('on'); cur.style.left = x + 'px'; cur.style.top = y + 'px'; },
    tap(x, y) { cur.classList.add('down'); setTimeout(() => cur.classList.remove('down'), 140);
      const r = document.createElement('div'); r.className = 'fx-ripple'; r.style.left = x + 'px'; r.style.top = y + 'px';
      layer.appendChild(r); setTimeout(() => r.remove(), 700); },
    hideCursor() { cur.classList.remove('on'); },
  };
})();
"""


class Director:
    def __init__(self, page, vertical):
        self.p = page
        self.v = vertical

    def fx(self, call, *args):
        self.p.evaluate(f"(a) => window.__fx.{call}(...a)", list(args))

    def wait(self, ms):
        self.p.wait_for_timeout(ms)

    def caption(self, html, hold=0):
        self.fx("caption", html)
        if hold:
            self.wait(hold)

    def card(self, html, hold):
        self.fx("card", html)
        self.wait(hold)

    def center(self, locator, ms=700):
        locator.evaluate("el => el.scrollIntoView({behavior: 'smooth', block: 'center'})")
        self.wait(ms)

    def top(self, locator, offset=70, ms=700):
        locator.evaluate(f"el => window.scrollTo({{top: el.getBoundingClientRect().top + scrollY - {offset}, behavior: 'smooth'}})")
        self.wait(ms)

    def spot(self, locator, color="#fff", pad=8):
        locator.evaluate(f"el => window.__fx.spot(el, '{color}', {pad})")

    def zoom(self, locator, scale, ms=950):
        locator.evaluate(f"el => window.__fx.zoom(el, {scale})")
        self.wait(ms)

    def unzoom(self, ms=900):
        self.fx("unzoom")
        self.wait(ms)

    def click(self, locator, move_ms=700):
        """Move the visible cursor to the element, then really click it."""
        box = locator.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        self.fx("cursorTo", x, y)
        self.wait(move_ms)
        self.fx("tap", x, y)
        locator.click()
        self.wait(150)

    def slide(self, locator, start, end, step, every=55):
        """Drag a React range input gradually (visible) from start to end."""
        box = locator.bounding_box()
        lo, hi = float(locator.get_attribute("min")), float(locator.get_attribute("max"))
        v = start
        n = 0
        while (step > 0 and v <= end + 1e-9) or (step < 0 and v >= end - 1e-9):
            x = box["x"] + box["width"] * (v - lo) / (hi - lo)
            self.fx("cursorTo", x, box["y"] + box["height"] / 2)
            locator.evaluate("""(el, v) => { const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                s.call(el, v); el.dispatchEvent(new Event('input', { bubbles: true })); }""", str(round(v, 4)))
            self.wait(every if n else 420)
            v += step
            n += 1


def fmt_pct(x):
    return f"{round(100 * x)}\u00a0%"


def flagged_share(arena_json, style):
    items = [i for i in arena_json["items"] if i["valid"] and i["style"] == style]
    return sum(i["suspicious"] for i in items) / len(items) if items else 0.0


def api(path, body=None):
    req = urllib.request.Request(URL.rstrip("/") + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


# ----------------------------------------------------------------------------- storyboard
def storyboard(d: Director, status: dict):
    p, v = d.p, d.v
    gen, det = status["generator"], status["classifier"]
    params = f"{gen['params']:,}".replace(",", ".")
    f1 = f"{det['metrics']['test']['f1']:.3f}".replace(".", ",")
    auc = f"{det['metrics']['test']['roc_auc']:.3f}".replace(".", ",")
    n_test = f"{det['metrics']['test']['n']:,}".replace(",", ".")

    # 1 — hook
    d.card(f"""<div class="fx-mono">PHANTOMLINK · AI vs AI</div>
        <h1>Entrené <span class="fx-r">2 IAs</span><br>para que se enfrentaran</h1>
        <p><span class="fx-r">🔴 Una inventa dominios.</span>{'<br>' if v else ' '}<span class="fx-b">🔵 La otra caza el phishing.</span></p>""", 2700)
    d.fx("card", None)
    d.wait(350)

    # 2 — the two models
    gen_card = p.locator(".panel").filter(has_text="IA generadora").first
    det_card = p.locator(".panel").filter(has_text="IA detectora").first
    d.caption("Las dos <span class='fx-g'>entrenadas desde cero</span><small>sin APIs · sin modelos preentrenados · en un portátil sin GPU</small>")
    d.center(gen_card, 500) if v else None
    d.spot(gen_card, "#ff3d5a")
    d.wait(1150)
    if v:
        d.center(det_card, 600)
    d.spot(det_card, "#3da5ff")
    d.wait(1100)
    d.fx("unspot")

    # 3 — first duel
    duel = p.get_by_role("button", name="INICIAR DUELO")
    if v:
        d.center(duel, 500)
    with p.expect_response(lambda r: r.url.endswith("/api/arena")) as resp:
        d.click(duel)
    r1 = resp.value.json()
    d.caption("<span class='fx-r'>🔴</span> Inventa 20 dominios <span class='fx-r'>letra a letra</span><small>ficticios: terminan en .test y nunca se visitan</small>")
    rows = p.locator(".rowin").first
    d.wait(300)
    if v:
        d.top(rows, 120, 900)
    else:
        d.top(p.get_by_text("Dominios generados → decisión del detector"), 80, 900)
        d.zoom(rows, 1.28)
    d.wait(800)
    first_flag = p.locator(".rowin").filter(has_text="SOSPECHOSO").filter(has_not_text="NO SOSPECHOSO").first
    flagged = sum(i["suspicious"] for i in r1["items"] if i["valid"])
    d.caption(f"<span class='fx-b'>🔵</span> La otra decide cuáles <span class='fx-b'>parecen phishing</span><small>marca {flagged} de {r1['summary']['valid']} · cada barra es su probabilidad real</small>")
    if first_flag.count():
        d.spot(first_flag, "#ff3d5a", 6)
    d.wait(2200)
    d.fx("unspot")
    d.unzoom(500)

    # 4 — temperature experiment
    p.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    d.wait(450)
    temp = p.locator("input[type=range]").first
    if v:
        d.center(temp, 500)
    d.caption("¿Y si le doy <span class='fx-r'>más creatividad</span>? 🌡️<small>temperatura 0,8 → 1,4</small>")
    d.slide(temp, 0.8, 1.4, 0.05)
    d.wait(300)
    if v:
        d.center(duel, 500)
    with p.expect_response(lambda r: r.url.endswith("/api/arena")) as resp:
        d.click(duel)
    r2 = resp.value.json()
    hist = p.get_by_text("Historial de rondas")
    d.wait(500)
    d.top(hist, 90 if not v else 70, 1000)
    a, b = flagged_share(r1, "phish"), flagged_share(r2, "phish")
    verb = "sube" if b > a else ("baja" if b < a else "se mantiene")
    d.caption(f"Dominios «estilo phish» que caza: <span class='fx-b'>{fmt_pct(a)}</span> → <span class='fx-r'>{fmt_pct(b)}</span><small>misma IA detectora · solo cambió la creatividad del generador ({verb})</small>")
    if not v:
        d.zoom(p.locator(".panel").filter(has_text="Historial de rondas").first, 1.15)
    d.wait(2600)
    d.unzoom(400)

    # 5 — inside the generator
    d.caption(None)
    d.click(p.get_by_role("button", name="Model Inspector"), 550)
    d.wait(400)
    gen_btn = p.get_by_role("button", name="Generar", exact=True)
    if v:
        d.center(gen_btn, 400)
    d.click(gen_btn)
    d.caption("Así elige <span class='fx-r'>cada letra</span><small>probabilidades reales del modelo, paso a paso</small>")
    word = p.locator(".font-mono.text-2xl").first
    d.wait(400)
    d.top(p.get_by_role("button", name="⏸ Pausa").first, 90 if not v else 60, 600)
    d.spot(word.locator(".."), "#ff3d5a", 6)
    d.wait(2900)
    d.fx("unspot")
    att = p.get_by_text("mapas de atención").first
    d.top(att, 70, 900)
    d.caption("Atención: <span class='fx-b'>dónde «mira»</span> la IA al escribir<small>cada cuadro es una cabeza del Transformer · valores reales</small>")
    d.spot(p.locator("svg.rounded").first.locator(".."), "#3da5ff", 6)
    d.wait(2100)
    d.fx("unspot")

    # 6 — live training
    d.caption(None)
    d.click(p.get_by_role("button", name="Training Lab"), 550)
    d.wait(300)
    fast = p.get_by_role("button", name="⚡ Fast").first
    if v:
        d.center(fast, 400)
    d.click(fast)
    train_btn = p.get_by_role("button", name="▶ Entrenar generador")
    if v:
        d.center(train_btn, 500)
    d.click(train_btn)
    d.caption("Y se <span class='fx-g'>entrena en directo</span><small>cada punto de la gráfica es una iteración real</small>")
    chart = p.get_by_text("loss (entropía cruzada)").first
    d.top(chart, 120 if not v else 60, 900)
    if not v:
        d.zoom(chart.locator(".."), 1.25)
    t0 = time.time()
    while time.time() - t0 < 25:  # until the job ends
        if p.locator("text=Guardado y activado").count():
            break
        d.wait(250)
    d.wait(700)
    d.unzoom(300)

    # 7 — outro
    d.caption(None)
    d.fx("hideCursor")
    d.card(f"""<div class="fx-mono">PHANTOMLINK</div>
        <div class="fx-stat">
          <div><b class="fx-r">{params}</b><span>parámetros · Transformer</span></div>
          <div><b class="fx-b">F1 {f1}</b><span>en {n_test} dominios de test</span></div>
          <div><b class="fx-g">AUC {auc}</b><span>sin ver la web, solo el nombre</span></div>
        </div>
        <p>Dos modelos pequeños, entrenados de verdad{'<br>' if v else ', '}y todo lo que pasa dentro, a la vista.</p>
        <div class="fx-mono">«no sospechoso» ≠ seguro · dominios ficticios .test</div>""", 3800)


# ----------------------------------------------------------------------------- pipeline
def ffmpeg_bin():
    for c in (os.environ.get("FFMPEG"), shutil.which("ffmpeg"), str(Path.home() / ".local/opt/ffmpeg/ffmpeg")):
        if c and Path(c).exists():
            return c
    raise SystemExit("ffmpeg not found: set FFMPEG=/path/to/ffmpeg")


class Screencast:
    """Full-resolution capture through the Chrome DevTools screencast.

    Playwright's built-in video recorder ignores device_scale_factor (it draws the CSS-size
    viewport in a corner of a grey canvas), so frames are taken from CDP instead, at real
    device pixels, each one with its own timestamp.
    """

    def __init__(self, page, size, folder: Path):
        self.folder = folder
        self.frames: list[tuple[float, Path]] = []
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._on_frame)
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92,
                                               "maxWidth": size["width"], "maxHeight": size["height"],
                                               "everyNthFrame": 1})

    def _on_frame(self, ev):
        path = self.folder / f"{len(self.frames):06d}.jpg"
        path.write_bytes(base64.b64decode(ev["data"]))
        self.frames.append((ev["metadata"]["timestamp"], path))
        try:
            self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
        except Exception:
            pass

    def stop(self):
        try:
            self.cdp.send("Page.stopScreencast")
        except Exception:
            pass

    def concat_list(self, t_start: float, t_end: float, out: Path) -> float:
        """ffmpeg concat-demuxer list with each frame shown until the next one arrives."""
        frames = sorted(self.frames)
        before = [f for f in frames if f[0] <= t_start]
        keep = ([before[-1]] if before else []) + [f for f in frames if t_start < f[0] < t_end]
        lines = []
        for i, (ts, path) in enumerate(keep):
            nxt = keep[i + 1][0] if i + 1 < len(keep) else t_end
            dur = max(0.001, nxt - max(ts, t_start))
            lines += [f"file '{path}'", f"duration {dur:.4f}"]
        lines.append(f"file '{keep[-1][1]}'")
        out.write_text("\n".join(lines) + "\n")
        return t_end - t_start


def record(fmt: str, chromium: str, ff: str) -> Path:
    spec = FORMATS[fmt]
    tmp = DOCS / f"_frames_{fmt}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    status = api("/api/status")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chromium, args=["--no-sandbox"])
        ctx = browser.new_context(viewport=spec["viewport"], device_scale_factor=spec["dsf"])
        page = ctx.new_page()
        page.goto(URL)
        page.evaluate("try { localStorage.clear() } catch (e) {}")
        page.reload()
        page.wait_for_selector("text=INICIAR DUELO")
        page.wait_for_timeout(700)
        page.evaluate(FX_JS)
        cast = Screencast(page, spec["size"], tmp)
        page.wait_for_timeout(300)
        t_start = time.time()
        d = Director(page, fmt == "9x16")
        storyboard(d, status)
        t_end = time.time()
        page.wait_for_timeout(200)
        cast.stop()
        ctx.close()
        browser.close()
    out = DOCS / f"demo_{fmt}.mp4"
    dur = cast.concat_list(t_start, t_end, tmp / "list.txt")
    w, h = spec["size"]["width"], spec["size"]["height"]
    bar = max(4, h // 270)
    # constant 30 fps, exact output size, thin progress bar, short fades, H.264 + faststart
    vf = (f"scale={w}:{h}:flags=lanczos,fps=30,"
          f"drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=0x1d2633@1:t=fill,"
          f"drawbox=x=0:y=ih-{bar}:w='iw*t/{dur:.3f}':h={bar}:color=0xff3d5a@1:t=fill,"
          f"fade=t=in:st=0:d=0.25,fade=t=out:st={max(0, dur - 0.5):.3f}:d=0.5")
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                    "-vf", vf, "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", "-an", str(out)],
                   check=True)
    n = len(cast.frames)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Saved {out}  ({dur:.1f} s, {w}x{h}, {n} captured frames ≈ {n / dur:.0f} fps)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FORMATS))
    args = ap.parse_args()
    chromium = os.environ.get("CHROMIUM") or shutil.which("chromium") or shutil.which("chromium-browser")
    ff = ffmpeg_bin()
    DOCS.mkdir(exist_ok=True)
    before = api("/api/status")["generator"]["id"]
    for fmt in ([args.only] if args.only else list(FORMATS)):
        try:
            record(fmt, chromium, ff)
        finally:
            # the storyboard trains a small model live: restore the original generator after
            # EVERY take, so the next format starts from exactly the same models
            api("/api/models/activate", {"kind": "generator", "id": before})


if __name__ == "__main__":
    main()
