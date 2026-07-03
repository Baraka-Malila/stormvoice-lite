"""Generate synthetic placeholder audio for 4 speakers.

Each speaker gets a distinct harmonic profile so the classifiers have
something discriminative to learn before real teammate recordings arrive.
Run once:
    python scripts/generate_placeholder_data.py
Replace data/recordings/<name>/*.wav with real recordings later — then
re-run the training notebooks.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
DURATION = 3.0
N_CLIPS = 60

# (fundamental Hz, first formant Hz, second formant Hz, vibrato rate Hz)
SPEAKERS: dict[str, tuple[float, float, float, float]] = {
    "baraka":   (130, 400, 1800, 4.8),
    "speaker2": (175, 620, 2200, 5.5),
    "speaker3": (220, 800, 2600, 6.2),
    "speaker4": (160, 500, 2050, 5.0),
}


def _make_clip(f0: float, f1: float, f2: float, vibrato: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)

    vib = 0.02 * np.sin(2 * np.pi * vibrato * t)
    sig = (
        0.50 * np.sin(2 * np.pi * f0 * (1 + vib) * t)
        + 0.25 * np.sin(2 * np.pi * f0 * 2 * t + rng.uniform(0, np.pi))
        + 0.15 * np.sin(2 * np.pi * f1 * t + rng.uniform(0, np.pi))
        + 0.08 * np.sin(2 * np.pi * f2 * t + rng.uniform(0, np.pi))
        + 0.03 * rng.standard_normal(len(t))
    )

    # Speech-like amplitude envelope
    frames = int(DURATION * 10)
    env_raw = np.abs(rng.standard_normal(frames)) ** 0.4
    env = np.interp(t, np.linspace(0, DURATION, frames), env_raw)

    sig = sig * env
    sig /= np.max(np.abs(sig) + 1e-8) * 1.4
    return sig.astype(np.float32)


def main(force: bool = False) -> None:
    root = Path(__file__).resolve().parent.parent / "data" / "recordings"
    root.mkdir(parents=True, exist_ok=True)

    for name, (f0, f1, f2, vib) in SPEAKERS.items():
        spk_dir = root / name
        spk_dir.mkdir(exist_ok=True)
        existing = list(spk_dir.glob("*.wav"))
        if existing and not force:
            print(f"  {name}: {len(existing)} clips already exist — skipping (use --force to overwrite)")
            continue
        print(f"  {name}: generating {N_CLIPS} clips…", end=" ", flush=True)
        for i in range(N_CLIPS):
            clip = _make_clip(f0, f1, f2, vib, seed=abs(hash(name)) + i)
            sf.write(spk_dir / f"{i:03d}.wav", clip, SR)
        print("done")

    print(f"\nPlaceholder data written to {root}")
    print("Replace with real recordings and re-run the training notebooks.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="Overwrite existing clips")
    main(p.parse_args().force)
