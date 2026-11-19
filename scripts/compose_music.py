"""Compose an original ambient-electronic soundtrack and add it to the demo videos.

The music is synthesised from scratch (additive oscillators, noise and a convolution reverb):
no samples, loops or melodies from third parties, so there are no third-party rights involved.
Deterministic: same video length -> same track.

    python scripts/compose_music.py docs/demo_16x9.mp4 docs/demo_9x16.mp4

Each video is muxed in place (the video stream is copied, not re-encoded).
Requires numpy, scipy and ffmpeg (FFMPEG=/path/to/ffmpeg if not on PATH).

Musical design (A minor, 100 BPM, i–VI–III–VII = Am–F–C–G, two bars per chord):
  intro    (title card)        warm pad only
  build 1  (two AIs, 1st duel) + sub bass, soft arpeggio, soft kick
  peak     (temperature test)  + off-beat hats, filter opens, arpeggio louder
  steady   (inspector, lab)    peak relaxes, pulse continues
  outro    (metrics card)      drums out, pad + arpeggio fade on the tonic
Section times are fractions of the 43.3 s storyboard, scaled to each video's length.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

SR = 44_100
BPM = 100
BEAT = 60 / BPM
BAR = 4 * BEAT
CHORDS = [  # (pad voicing in MIDI, bass root)
    ([57, 60, 64, 69], 45),  # Am
    ([53, 57, 60, 65], 41),  # F
    ([55, 60, 64, 67], 48),  # C
    ([55, 59, 62, 67], 43),  # G
]
# storyboard landmarks of the 43.3 s cut (title, duel, temperature, inspector, training, outro)
REF = 43.3
SECTIONS = {"build1": 3.2, "kick": 6.4, "peak": 13.0, "peak_end": 21.5, "outro": 38.3}


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def ramp(t, t0, t1, a=0.0, b=1.0):
    """Linear ramp from a (before t0) to b (after t1)."""
    x = np.clip((t - t0) / max(t1 - t0, 1e-6), 0, 1)
    return a + (b - a) * x


def smooth(x):
    return x * x * (3 - 2 * x)


def compose(duration: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * SR) + SR * 3  # tail for the reverb, trimmed later
    t = np.arange(n) / SR
    k = duration / REF
    S = {name: v * k for name, v in SECTIONS.items()}

    # --- global automation curves -------------------------------------------------------
    build = smooth(ramp(t, S["build1"], S["peak"]))                      # 0 -> 1
    peak = smooth(ramp(t, S["peak"] - 1.5, S["peak"] + 2.0)) * (1 - smooth(ramp(t, S["peak_end"], S["peak_end"] + 4.0)))
    outro = smooth(ramp(t, S["outro"] - 0.5, S["outro"] + 1.5))           # 0 -> 1
    post = smooth(ramp(t, S["peak_end"], S["peak_end"] + 4.0))           # after the peak
    brightness = 2.0 + 3.5 * build + 4.0 * peak - 2.5 * outro             # harmonics roll-off

    pad = np.zeros((n, 2))
    bass = np.zeros(n)
    arp = np.zeros((n, 2))
    drums = np.zeros((n, 2))

    # --- pad: band-limited detuned saws, harmonic roll-off follows `brightness` -----------
    seg = 2 * BAR
    n_seg = int(np.ceil(duration / seg)) + 1
    att, rel = 0.9, 1.6
    for s in range(n_seg):
        notes, root = CHORDS[s % len(CHORDS)]
        if t[0] + s * seg > duration + 0.5:
            break
        if s * seg >= S["outro"] + 0.5:          # resolve the outro on the tonic (Am)
            notes, root = CHORDS[0]
        a0 = int(s * seg * SR)
        a1 = min(n, int(((s + 1) * seg + rel) * SR))
        tt = t[a0:a1] - s * seg
        env = np.minimum(1, tt / att) * np.clip(((seg + rel) - tt) / rel, 0, 1)
        bright = brightness[a0:a1]
        for vi, m in enumerate(notes):
            for det, pan in ((-0.07, 0.2), (0.0, 0.5), (0.07, 0.8)):
                f = midi(m + det)
                ph = rng.uniform(0, 2 * np.pi)
                wave = np.zeros_like(tt)
                for h in range(1, 15):
                    if h * f > 6000:
                        break
                    wave += np.sin(2 * np.pi * h * f * tt + h * ph) / h * np.exp(-(h - 1) / bright)
                wave *= env * (0.9 if vi else 1.0)
                pad[a0:a1, 0] += wave * (1 - pan)
                pad[a0:a1, 1] += wave * pan
        # sub bass: sine + a little 2nd harmonic, pumping on each beat (sidechain feel)
        fb = midi(root - 12 if root > 44 else root)
        beat_phase = (tt % BEAT) / BEAT
        pump = 0.55 + 0.45 * (1 - np.exp(-beat_phase * BEAT / 0.12))
        b = (np.sin(2 * np.pi * fb * tt) + 0.25 * np.sin(4 * np.pi * fb * tt)) * env * pump
        bass[a0:a1] += b

    # --- arpeggio: plucked notes on 8ths, chord tones up an octave -------------------------
    eighth = BEAT / 2
    pattern = [0, 1, 2, 3, 2, 1, 3, 2]
    step = 0
    pos = S["build1"]
    while pos < duration - 0.3:
        s = int(pos // seg)
        notes, _ = CHORDS[0] if pos >= S["outro"] + 0.5 else CHORDS[s % len(CHORDS)]
        m = notes[pattern[step % len(pattern)]] + 12
        f = midi(m)
        L = int(0.9 * SR)
        a0 = int(pos * SR)
        a1 = min(n, a0 + L)
        tt = np.arange(a1 - a0) / SR
        env = np.minimum(1, tt / 0.004) * np.exp(-tt / (0.22 + 0.1 * peak[a0]))
        note = (np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(4 * np.pi * f * tt) * np.exp(-tt / 0.08)
                + 0.12 * np.sin(6 * np.pi * f * tt) * np.exp(-tt / 0.05)) * env
        accent = 1.0 if step % 4 == 0 else 0.72
        level = accent * (0.25 + 0.45 * build[a0] + 0.5 * peak[a0]) * (1 - 0.3 * post[a0]) * (1 - 0.6 * outro[a0])
        pan = 0.3 if step % 2 else 0.7
        arp[a0:a1, 0] += note * level * (1 - pan)
        arp[a0:a1, 1] += note * level * pan
        step += 1
        pos += eighth

    # --- drums: soft kick on beats, off-beat hats only around the peak ---------------------
    hp_noise = np.diff(rng.standard_normal(int(0.08 * SR) + 1))
    pos = S["kick"]
    beat_i = 0
    while pos < S["outro"] + BEAT:
        a0 = int(pos * SR)
        L = int(0.35 * SR)
        a1 = min(n, a0 + L)
        tt = np.arange(a1 - a0) / SR
        fk = 48 + 85 * np.exp(-tt / 0.035)
        kick = np.sin(2 * np.pi * np.cumsum(fk) / SR) * np.exp(-tt / 0.18) * np.minimum(1, tt / 0.002)
        lvl = 0.55 * smooth(ramp(pos, S["kick"], S["kick"] + 3)) * (1 - outro[a0]) * (0.6 + 0.4 * peak[a0]) * (1 - 0.2 * post[a0])
        drums[a0:a1] += (kick * lvl)[:, None]
        hat_lvl = 0.16 * peak[min(n - 1, int((pos + eighth) * SR))]
        if hat_lvl > 0.005:
            h0 = int((pos + eighth) * SR)
            h1 = min(n, h0 + len(hp_noise))
            th = np.arange(h1 - h0) / SR
            hat = hp_noise[: h1 - h0] * np.exp(-th / 0.025) * hat_lvl
            drums[h0:h1, 0] += hat * (0.6 if beat_i % 2 else 0.4)
            drums[h0:h1, 1] += hat * (0.4 if beat_i % 2 else 0.6)
        beat_i += 1
        pos += BEAT

    # --- reverb (stereo convolution with decaying, darkened noise) -------------------------
    ir_len = int(2.6 * SR)
    ti = np.arange(ir_len) / SR
    ir = rng.standard_normal((ir_len, 2)) * np.exp(-ti / 0.75)[:, None]
    kernel = np.ones(12) / 12
    ir = np.stack([np.convolve(ir[:, c], kernel, mode="same") for c in range(2)], axis=1)
    ir[: int(0.02 * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum(axis=0))

    def verb(x, wet):
        y = np.stack([fftconvolve(x[:, c], ir[:, c])[:n] for c in range(2)], axis=1)
        return x * (1 - wet) + y * wet

    pad_level = 0.11 * ((0.55 + 0.35 * build + 0.35 * peak) * (1 - 0.15 * post) * (1 - 0.35 * outro))[:, None]
    mix = (verb(pad * pad_level, 0.45)
           + bass[:, None] * (0.22 * smooth(ramp(t, S["build1"], S["build1"] + 2.5)) * (1 - 0.85 * outro))[:, None]
           + verb(arp * 0.11, 0.35)
           + drums * 0.5)

    # fades and soft limiting
    fade_in = smooth(ramp(t, 0, 2.0))
    fade_out = 1 - smooth(ramp(t, duration - 2.2, duration))
    mix *= (fade_in * fade_out)[:, None]
    mix = mix[: int(duration * SR)]
    mix = np.tanh(mix * 1.4) / np.tanh(1.4)
    mix /= max(1e-9, np.abs(mix).max()) / 0.89  # -1 dBFS peak before loudness normalisation
    return mix.astype(np.float32)


def ff_bin(name="ffmpeg"):
    for c in (os.environ.get(name.upper()), shutil.which(name), str(Path.home() / f".local/opt/ffmpeg/{name}")):
        if c and Path(c).exists():
            return c
    raise SystemExit(f"{name} not found")


def duration_of(video: Path) -> float:
    out = subprocess.run([ff_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(video)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def integrated_loudness(wav: Path) -> float:
    """EBU R128 integrated loudness (LUFS) measured by ffmpeg."""
    err = subprocess.run([ff_bin(), "-hide_banner", "-i", str(wav), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    summary = err[err.rfind("Summary:"):]
    return float(summary.split("I:")[1].split("LUFS")[0])


def add_music(video: Path, loudness: float = -21.0) -> None:
    dur = duration_of(video)
    audio = compose(dur)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "music.wav"
        wavfile.write(wav, SR, audio)
        # one fixed gain (linear): keeps the intro / peak / outro arc intact, unlike a
        # dynamic loudness normaliser that would flatten it
        gain = loudness - integrated_loudness(wav)
        out = Path(tmp) / video.name
        subprocess.run([ff_bin(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav),
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                        "-af", f"volume={gain:.2f}dB,alimiter=limit=0.79:level=false",
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{dur:.3f}",
                        "-movflags", "+faststart", str(out)], check=True)
        shutil.move(str(out), video)
    print(f"Music added to {video} ({dur:.1f} s, target {loudness} LUFS)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for v in sys.argv[1:]:
        add_music(Path(v))
