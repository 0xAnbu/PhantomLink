"""Record a ~40 s demo video of the running app following the storyboard in the docs.

Dev-only tool:  pip install playwright   (uses a local Chromium; set CHROMIUM=/path if needed)
    ./run.sh &   then   python scripts/record_demo.py  -> docs/demo.webm (+ demo.mp4 if ffmpeg exists)
Everything shown is produced live by the models; the script only clicks and scrolls.
"""
import os
import shutil
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = os.environ.get("PHANTOMLINK_URL", "http://127.0.0.1:8000/")
OUT = Path(__file__).resolve().parents[1] / "docs"
SIZE = {"width": 1280, "height": 720}


def set_range(page, locator, value):
    """Set a React-controlled <input type=range> like a user would."""
    locator.evaluate(
        """(el, v) => {
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
            setter.call(el, v); el.dispatchEvent(new Event('input', { bubbles: true }));
        }""", str(value))


def smooth_scroll(page, y, ms=900):
    page.evaluate(f"window.scrollTo({{top: {y}, behavior: 'smooth'}})")
    page.wait_for_timeout(ms)


def main():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM") or shutil.which("chromium"),
                                    args=["--no-sandbox"])
        ctx = browser.new_context(viewport=SIZE, record_video_dir=str(OUT / "_raw"), record_video_size=SIZE)
        page = ctx.new_page()
        page.goto(URL)
        page.evaluate("localStorage.clear()")
        page.reload()
        page.wait_for_timeout(3500)                                        # 0-4 s: the two AIs

        duel = page.get_by_role("button", name="INICIAR DUELO")
        duel.click()
        page.wait_for_timeout(1500)
        smooth_scroll(page, 520, 2500)                                     # 4-9 s: names + decisions
        smooth_scroll(page, 0, 900)

        temp = page.locator("input[type=range]").first                     # 9-17 s: temperature
        for v in (1.0, 1.2, 1.4):
            set_range(page, temp, v)
            page.wait_for_timeout(300)
        duel.click()
        page.wait_for_timeout(1200)
        smooth_scroll(page, 520, 2200)
        smooth_scroll(page, 1500, 1800)                                    # round history

        page.get_by_role("button", name="Model Inspector").click()         # 17-29 s: inspector
        page.evaluate("window.scrollTo(0,0)")
        page.wait_for_timeout(600)
        page.get_by_role("button", name="Generar", exact=True).click()
        page.wait_for_timeout(5200)                                        # autoplay char by char
        smooth_scroll(page, 640, 3000)                                     # attention maps
        smooth_scroll(page, 0, 600)

        page.get_by_role("button", name="Training Lab").click()            # 29-42 s: training
        page.wait_for_timeout(600)
        page.get_by_role("button", name="⚡ Fast").first.click()
        page.get_by_role("button", name="▶ Entrenar generador").click()
        page.wait_for_timeout(1000)
        smooth_scroll(page, 250, 800)
        page.wait_for_timeout(9000)
        ctx.close()
        browser.close()

    raw = sorted((OUT / "_raw").glob("*.webm"), key=lambda f: f.stat().st_mtime)[-1]
    final = OUT / "demo.webm"
    raw.replace(final)
    shutil.rmtree(OUT / "_raw", ignore_errors=True)
    print(f"Saved {final}")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(final), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                        str(OUT / "demo.mp4")], check=True)
        print(f"Saved {OUT / 'demo.mp4'}")


if __name__ == "__main__":
    main()
