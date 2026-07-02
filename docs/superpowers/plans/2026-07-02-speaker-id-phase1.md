# Speaker ID Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the end-to-end Phase 1 speaker-identification pipeline — data ingest → MFCC + log-mel features → SVM baseline + small CNN → evaluation with confusion matrices → live-demo web app — using LibriSpeech placeholder speakers so it runs today with only Baraka's own voice.

**Architecture:** Six Jupyter notebooks orchestrate the ML lifecycle (ingest → features → SVM → CNN → eval). Three small `.py` modules hold reusable helpers imported by the notebooks and the FastAPI server: `features/mfcc.py`, `features/spectrogram.py`, `models/cnn.py`. A single-file FastAPI app (`scripts/serve.py`) loads both trained artifacts once at startup and exposes `/predict`. Frontend is a fork of the bongoSTEM shell.

**Tech Stack:** Python 3.10+, librosa, scikit-learn, PyTorch, torchaudio (for LibriSpeech download), FastAPI + uvicorn, matplotlib, Plotly.js, vanilla JS + `MediaRecorder`. Tests use pytest; notebook smoke tests use `jupyter nbconvert --execute`.

## Global Constraints

- **Python 3.10+** — target for venv.
- **300-line limit** on `.py` files (from `CLAUDE.md`); notebooks exempt.
- **One responsibility per `.py` file** — `features/mfcc.py` only extracts MFCC, `models/cnn.py` only defines the architecture, etc.
- **Sample rate: 16 kHz mono** everywhere. Anything else gets resampled at ingest.
- **Clip length: 3.0 seconds** exact, non-overlapping windows.
- **Fail fast on missing recordings** — clear error, no silent defaults.
- **Split policy: 80/10/10 grouped by source file, stratified by speaker, seed = 42.**
- **Directory layout matches the spec** — do not restructure.
- **Do NOT modify anything under `~/bongoSTEM/`** — copy assets, never edit in place.
- **Frequent commits** — one commit per finished task minimum, more when a task has independently-reviewable sub-steps.

## File Structure

**Created in this plan:**

- `requirements.txt` — pinned deps (Task 1).
- `.gitignore` additions (Task 1).
- `features/__init__.py`, `features/mfcc.py`, `features/spectrogram.py` (Tasks 2, 3).
- `models/__init__.py`, `models/cnn.py` (Task 4).
- `tests/__init__.py`, `tests/test_mfcc.py`, `tests/test_spectrogram.py`, `tests/test_cnn.py`, `tests/test_serve.py` (Tasks 2, 3, 4, 10).
- `notebooks/01_ingest.ipynb` through `notebooks/06_evaluate_and_compare.ipynb` (Tasks 5, 6, 7, 8, 9).
- `scripts/serve.py` (Task 10).
- `frontend/static/` (forked from bongoSTEM), `frontend/templates/index.html` (Task 11).
- `docs/design.md`, `docs/slides.md`, `docs/figures/` (Tasks 6, 9, 12).
- `README.md` — rewritten with quickstart (Task 12).

**Responsibility per file:**

- `features/mfcc.py` — `extract_mfcc(clip_path) -> np.ndarray` of shape `(120,)`. Nothing else.
- `features/spectrogram.py` — `extract_logmel(clip_path) -> np.ndarray` of shape `(64, 300)`. Nothing else.
- `models/cnn.py` — `SpeakerCNN(nn.Module)` architecture only. No training loop, no I/O.
- `scripts/serve.py` — FastAPI app: startup loads artifacts, `/`, `/static/*`, `/predict`. Reuses `features/*` helpers.
- Notebooks — one story each, cells map to slide sections.

---

## Task 1: Environment and scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `features/__init__.py`, `models/__init__.py`, `scripts/__init__.py`, `tests/__init__.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: a working venv with all deps installed and the empty package skeleton importable.

- [ ] **Step 1: Create `requirements.txt`**

```
librosa>=0.10.0
scikit-learn>=1.4.0
torch>=2.1.0
torchaudio>=2.1.0
sounddevice>=0.4.6
soundfile>=0.12.0
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
matplotlib>=3.8.0
numpy>=1.26.0
scipy>=1.11.0
pandas>=2.1.0
joblib>=1.3.0
jupyter>=1.0.0
jupytext>=1.16.0
pytest>=7.4.0
httpx>=0.26.0
python-multipart>=0.0.9
```

- [ ] **Step 2: Create the venv and install deps**

Run:
```bash
cd /home/cyberpunk/speech-recognition
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Expected: install succeeds, no ResolutionImpossible errors.

- [ ] **Step 3: Update `.gitignore` — append these lines to the existing file**

```
venv/
data/recordings/
data/clips/
models/artifacts/
docs/figures/*.png
docs/figures/*.jpg
.ipynb_checkpoints/
__pycache__/
*.pyc
.pytest_cache/
```

The `docs/figures/*.png` (rather than `docs/figures/`) form keeps the directory
committable via `.gitkeep` while ignoring the generated images themselves.

- [ ] **Step 4: Create empty package init files**

Create `features/__init__.py`, `models/__init__.py`, `scripts/__init__.py`, `tests/__init__.py` each with exactly this content:

```python
```

(Empty file. `touch` works.)

- [ ] **Step 5: Verify Python imports resolve**

Run:
```bash
source venv/bin/activate
python -c "import librosa, torch, sklearn, fastapi; print('OK')"
```
Expected: prints `OK`, no ImportError.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore features/__init__.py models/__init__.py scripts/__init__.py tests/__init__.py
git commit -m "chore: env scaffolding for speaker-id Phase 1"
```

---

## Task 2: MFCC feature helper (`features/mfcc.py`)

**Files:**
- Create: `features/mfcc.py`
- Create: `tests/test_mfcc.py`

**Interfaces:**
- Consumes: any 16 kHz mono `.wav` file path (str or `pathlib.Path`).
- Produces: `extract_mfcc(clip_path) -> np.ndarray` returning shape `(120,)` dtype `float32`. Consumed by Tasks 7 (SVM training) and 10 (server).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mfcc.py`:
```python
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

from features.mfcc import extract_mfcc


@pytest.fixture
def sine_wav(tmp_path: Path) -> Path:
    """3-second 440 Hz sine at 16 kHz mono. Deterministic input for shape checks."""
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "sine.wav"
    sf.write(path, y.astype(np.float32), sr)
    return path


def test_extract_mfcc_returns_fixed_120_dim_vector(sine_wav: Path):
    vec = extract_mfcc(str(sine_wav))
    assert vec.shape == (120,), f"expected (120,), got {vec.shape}"
    assert vec.dtype == np.float32


def test_extract_mfcc_is_deterministic(sine_wav: Path):
    a = extract_mfcc(str(sine_wav))
    b = extract_mfcc(str(sine_wav))
    np.testing.assert_array_equal(a, b)


def test_extract_mfcc_handles_shorter_clip(tmp_path: Path):
    """A 1.5s clip should still produce a (120,) vector — librosa handles short input."""
    sr = 16000
    y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, int(sr * 1.5), endpoint=False))
    path = tmp_path / "short.wav"
    sf.write(path, y.astype(np.float32), sr)
    vec = extract_mfcc(str(path))
    assert vec.shape == (120,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/test_mfcc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.mfcc'`.

- [ ] **Step 3: Implement `features/mfcc.py`**

Create `features/mfcc.py`:
```python
"""MFCC + delta + delta-delta extractor collapsed to a fixed-length descriptor.

The SVM classifier consumes a fixed-size input, so the (60, T) frame-level
matrix is reduced by taking mean and std across time -> (120,).
"""
from __future__ import annotations

import numpy as np
import librosa

_SR = 16000
_N_MFCC = 20
_N_FFT = 512
_HOP_LENGTH = 160  # 10 ms
_WIN_LENGTH = 400  # 25 ms


def extract_mfcc(clip_path: str) -> np.ndarray:
    y, _ = librosa.load(clip_path, sr=_SR, mono=True)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=_SR,
        n_mfcc=_N_MFCC,
        n_fft=_N_FFT,
        hop_length=_HOP_LENGTH,
        win_length=_WIN_LENGTH,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack([mfcc, delta, delta2])  # (60, T)
    descriptor = np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)])  # (120,)
    return descriptor.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mfcc.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add features/mfcc.py tests/test_mfcc.py
git commit -m "feat(features): MFCC extractor with delta and delta-delta"
```

---

## Task 3: Log-mel spectrogram helper (`features/spectrogram.py`)

**Files:**
- Create: `features/spectrogram.py`
- Create: `tests/test_spectrogram.py`

**Interfaces:**
- Consumes: any 16 kHz mono `.wav` file path.
- Produces: `extract_logmel(clip_path) -> np.ndarray` returning shape `(64, 300)` dtype `float32`. Consumed by Tasks 8 (CNN training) and 10 (server).

- [ ] **Step 1: Write the failing test**

Create `tests/test_spectrogram.py`:
```python
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

from features.spectrogram import extract_logmel


@pytest.fixture
def sine_wav_3s(tmp_path: Path) -> Path:
    sr = 16000
    y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3.0, sr * 3, endpoint=False))
    path = tmp_path / "sine.wav"
    sf.write(path, y.astype(np.float32), sr)
    return path


def test_extract_logmel_returns_64_by_300(sine_wav_3s: Path):
    spec = extract_logmel(str(sine_wav_3s))
    assert spec.shape == (64, 300), f"expected (64, 300), got {spec.shape}"
    assert spec.dtype == np.float32


def test_extract_logmel_is_normalized(sine_wav_3s: Path):
    """Per-clip normalization -> mean ~0, std ~1."""
    spec = extract_logmel(str(sine_wav_3s))
    assert abs(spec.mean()) < 1e-3
    assert abs(spec.std() - 1.0) < 1e-3


def test_extract_logmel_pads_short_clip(tmp_path: Path):
    """A 1.5s clip -> (64, 300) with zero-pad on the right."""
    sr = 16000
    y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, int(sr * 1.5), endpoint=False))
    path = tmp_path / "short.wav"
    sf.write(path, y.astype(np.float32), sr)
    spec = extract_logmel(str(path))
    assert spec.shape == (64, 300)


def test_extract_logmel_crops_long_clip(tmp_path: Path):
    """A 5s clip -> (64, 300), cropped from the middle."""
    sr = 16000
    y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 5.0, sr * 5, endpoint=False))
    path = tmp_path / "long.wav"
    sf.write(path, y.astype(np.float32), sr)
    spec = extract_logmel(str(path))
    assert spec.shape == (64, 300)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_spectrogram.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.spectrogram'`.

- [ ] **Step 3: Implement `features/spectrogram.py`**

Create `features/spectrogram.py`:
```python
"""Log-mel spectrogram at (64 mel bins, 300 time frames) — CNN input."""
from __future__ import annotations

import numpy as np
import librosa

_SR = 16000
_N_MELS = 64
_N_FFT = 512
_HOP_LENGTH = 160  # 10 ms  ->  300 frames per 3 s
_TARGET_FRAMES = 300
_TARGET_SAMPLES = _SR * 3  # 3 s at 16 kHz


def _fit_to_3s(y: np.ndarray) -> np.ndarray:
    """Center-crop long input, zero-pad short input to exactly 3 s."""
    n = y.shape[0]
    if n == _TARGET_SAMPLES:
        return y
    if n > _TARGET_SAMPLES:
        start = (n - _TARGET_SAMPLES) // 2
        return y[start : start + _TARGET_SAMPLES]
    out = np.zeros(_TARGET_SAMPLES, dtype=y.dtype)
    out[:n] = y
    return out


def extract_logmel(clip_path: str) -> np.ndarray:
    y, _ = librosa.load(clip_path, sr=_SR, mono=True)
    y = _fit_to_3s(y)
    mel = librosa.feature.melspectrogram(
        y=y, sr=_SR, n_mels=_N_MELS, n_fft=_N_FFT, hop_length=_HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # (64, ~301)
    log_mel = log_mel[:, :_TARGET_FRAMES]  # trim any off-by-one
    # per-clip normalize
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spectrogram.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add features/spectrogram.py tests/test_spectrogram.py
git commit -m "feat(features): log-mel spectrogram extractor (64x300)"
```

---

## Task 4: CNN architecture (`models/cnn.py`)

**Files:**
- Create: `models/cnn.py`
- Create: `tests/test_cnn.py`

**Interfaces:**
- Consumes: `torch.Tensor` batch of shape `(N, 1, 64, 300)`.
- Produces: `SpeakerCNN(num_classes: int).forward(x) -> Tensor` of shape `(N, num_classes)` returning **raw logits** (softmax applied at loss / inference time). Consumed by Tasks 8 (training) and 10 (server).

- [ ] **Step 1: Write the failing test**

Create `tests/test_cnn.py`:
```python
import torch
import pytest

from models.cnn import SpeakerCNN


def test_forward_returns_logits_of_correct_shape():
    model = SpeakerCNN(num_classes=4)
    x = torch.randn(2, 1, 64, 300)  # batch of 2 log-mels
    out = model(x)
    assert out.shape == (2, 4)


def test_supports_different_num_classes():
    model = SpeakerCNN(num_classes=10)
    x = torch.randn(3, 1, 64, 300)
    out = model(x)
    assert out.shape == (3, 10)


def test_forward_is_differentiable():
    model = SpeakerCNN(num_classes=4)
    x = torch.randn(1, 1, 64, 300, requires_grad=False)
    y = torch.tensor([1])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    # every parameter should have a grad
    for p in model.parameters():
        assert p.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cnn.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.cnn'`.

- [ ] **Step 3: Implement `models/cnn.py`**

Create `models/cnn.py`:
```python
"""Small speaker-ID CNN operating on 64x300 log-mel spectrograms."""
from __future__ import annotations

import torch
from torch import nn


class SpeakerCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cnn.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add models/cnn.py tests/test_cnn.py
git commit -m "feat(models): small speaker-ID CNN architecture"
```

---

## Task 5: Ingest notebook (`notebooks/01_ingest.ipynb`)

**Files:**
- Create: `notebooks/01_ingest.ipynb`

**Interfaces:**
- Consumes: raw audio in `data/recordings/<speaker>/*` (any codec ffmpeg reads).
- Produces: `data/clips/<speaker>/<speaker>_<idx>.wav` (3 s each, 16 kHz mono), `data/clips/manifest.csv`, `data/clips/splits.csv`.

Create the notebook by making one cell per Python block below. The recommended workflow: `jupyter notebook notebooks/01_ingest.ipynb` → paste each block into a fresh code cell in order.

- [ ] **Step 1: Cell 1 — imports and paths**

```python
import os
import shutil
import random
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

REPO_ROOT = Path.cwd().parent
RECORDINGS = REPO_ROOT / "data" / "recordings"
CLIPS = REPO_ROOT / "data" / "clips"
CLIP_DURATION_S = 3.0
SR = 16000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
```

- [ ] **Step 2: Cell 2 — download LibriSpeech placeholders**

```python
# The spec calls for placeholder speakers so the pipeline is testable before
# teammate data lands. LibriSpeech dev-clean has ~40 speakers, download is
# ~340 MB, no auth. We pick three speakers, extract ~10 min each, and drop
# the files into data/recordings/standin_{a,b,c}/.
import torchaudio

LIBRISPEECH_ROOT = REPO_ROOT / "data" / "_librispeech"
LIBRISPEECH_ROOT.mkdir(parents=True, exist_ok=True)

ds = torchaudio.datasets.LIBRISPEECH(
    root=str(LIBRISPEECH_ROOT), url="dev-clean", download=True
)

# Group utterances by speaker id
by_speaker: dict[int, list[int]] = {}
for i in range(len(ds)):
    _wav, _sr, _txt, spk_id, _chap, _utt = ds.get_metadata(i)
    by_speaker.setdefault(spk_id, []).append(i)

# Sort by count desc; pick the three most talkative distinct speakers
top = sorted(by_speaker.items(), key=lambda kv: len(kv[1]), reverse=True)[:3]
print("Chosen standin speaker ids:", [spk for spk, _ in top])

STANDIN_NAMES = ["standin_a", "standin_b", "standin_c"]
for name, (spk_id, idxs) in zip(STANDIN_NAMES, top):
    out_dir = RECORDINGS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    total_sec = 0.0
    for k, i in enumerate(idxs):
        wav, sr, _txt, _spk, _chap, _utt = ds[i]
        y = wav.numpy().squeeze()
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        dur = len(y) / SR
        sf.write(out_dir / f"{name}_{k:03d}.wav", y, SR)
        total_sec += dur
        if total_sec >= 600:  # ~10 min per speaker
            break
    print(f"{name}: {total_sec:.1f}s from librispeech speaker {spk_id}")
```

- [ ] **Step 3: Cell 3 — verify Baraka's own recordings are present**

```python
BARAKA_DIR = RECORDINGS / "baraka"
if not BARAKA_DIR.exists() or not any(BARAKA_DIR.iterdir()):
    raise FileNotFoundError(
        f"Drop phone recordings into {BARAKA_DIR} before running this notebook."
    )
print("baraka raw files:", sorted(p.name for p in BARAKA_DIR.iterdir()))
```

- [ ] **Step 4: Cell 4 — VAD + chop helper**

```python
def load_and_normalize(path: Path) -> np.ndarray:
    """Decode any codec -> 16 kHz mono float32, trim edge silence."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    return y_trim


def chop_to_clips(y: np.ndarray, window_s: float = CLIP_DURATION_S) -> list[np.ndarray]:
    """Non-overlapping 3-second windows. Drop tail shorter than window."""
    win = int(window_s * SR)
    n_clips = len(y) // win
    return [y[i * win : (i + 1) * win] for i in range(n_clips)]


def clip_is_speech(clip: np.ndarray, energy_threshold: float = 0.005) -> bool:
    """Drop clips that are mostly silence (RMS below threshold)."""
    return float(np.sqrt(np.mean(clip ** 2))) > energy_threshold
```

- [ ] **Step 5: Cell 5 — run ingest and build the manifest**

```python
CLIPS.mkdir(parents=True, exist_ok=True)
manifest_rows: list[dict] = []

speakers = sorted(p.name for p in RECORDINGS.iterdir() if p.is_dir() and not p.name.startswith("_"))
print("speakers found:", speakers)

for speaker in speakers:
    src = RECORDINGS / speaker
    dst = CLIPS / speaker
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    clip_idx = 0
    for source_file in sorted(src.iterdir()):
        if source_file.name.startswith("."):
            continue
        try:
            y = load_and_normalize(source_file)
        except Exception as e:
            print(f"  skip {source_file.name}: {e}")
            continue

        for chunk in chop_to_clips(y):
            if not clip_is_speech(chunk):
                continue
            clip_name = f"{speaker}_{clip_idx:04d}.wav"
            clip_path = dst / clip_name
            sf.write(clip_path, chunk, SR)
            manifest_rows.append({
                "clip_path": str(clip_path.relative_to(REPO_ROOT)),
                "speaker": speaker,
                "source_file": source_file.name,
                "duration": CLIP_DURATION_S,
            })
            clip_idx += 1
    print(f"  {speaker}: {clip_idx} clips")

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(CLIPS / "manifest.csv", index=False)
manifest.groupby("speaker").size()
```

- [ ] **Step 6: Cell 6 — grouped-by-source train/val/test split**

```python
# Splits are grouped by source_file so no two clips from the same original
# recording end up in different splits (prevents leakage between adjacent
# 3s windows).
rng = random.Random(SEED)

split_rows: list[dict] = []
for speaker, group in manifest.groupby("speaker"):
    files = group["source_file"].unique().tolist()
    rng.shuffle(files)
    n = len(files)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    train_files = set(files[:n_train])
    val_files = set(files[n_train : n_train + n_val])
    test_files = set(files[n_train + n_val :])

    for _, row in group.iterrows():
        sf_name = row["source_file"]
        if sf_name in train_files:
            split = "train"
        elif sf_name in val_files:
            split = "val"
        else:
            split = "test"
        split_rows.append({"clip_path": row["clip_path"], "split": split})

splits = pd.DataFrame(split_rows)
splits.to_csv(CLIPS / "splits.csv", index=False)
print(splits.groupby("split").size())
```

- [ ] **Step 7: Cell 7 — quick sanity plot of one clip per speaker**

```python
fig, axes = plt.subplots(len(speakers), 1, figsize=(10, 2 * len(speakers)))
if len(speakers) == 1:
    axes = [axes]
for ax, speaker in zip(axes, speakers):
    sample = manifest[manifest["speaker"] == speaker].iloc[0]
    y, _ = librosa.load(REPO_ROOT / sample["clip_path"], sr=SR)
    ax.plot(np.linspace(0, 3, len(y)), y)
    ax.set_title(speaker)
    ax.set_ylabel("amp")
axes[-1].set_xlabel("s")
plt.tight_layout()
```

- [ ] **Step 8: Run the notebook end-to-end (smoke test)**

Drop at least one `.m4a` (or any audio) file into `data/recordings/baraka/` first (a short WhatsApp voice note is fine for the smoke run). Then:

```bash
source venv/bin/activate
cd notebooks
jupyter nbconvert --to notebook --execute 01_ingest.ipynb --output 01_ingest.ipynb
```
Expected: notebook runs to the last cell without error. `data/clips/manifest.csv` and `data/clips/splits.csv` exist. Per-speaker clip counts printed.

If the LibriSpeech download times out, re-run — torchaudio resumes.

- [ ] **Step 9: Commit**

```bash
git add notebooks/01_ingest.ipynb
git commit -m "feat(notebooks): ingest pipeline (raw -> clips + manifest + splits)"
```

---

## Task 6: Feature-visualization notebooks (`02_features_mfcc.ipynb`, `03_features_melspec.ipynb`)

**Files:**
- Create: `notebooks/02_features_mfcc.ipynb`
- Create: `notebooks/03_features_melspec.ipynb`
- Create: `docs/figures/` (directory)

**Interfaces:**
- Consumes: `data/clips/manifest.csv`, plus `features/mfcc.py` and `features/spectrogram.py`.
- Produces: `docs/figures/mfcc_by_speaker.png`, `docs/figures/melspec_by_speaker.png` (slide material). No trained models yet.

- [ ] **Step 1: Create `docs/figures/` directory**

```bash
mkdir -p docs/figures
```

- [ ] **Step 2: Write `notebooks/02_features_mfcc.ipynb` — Cell 1 (setup)**

```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))  # make features/ importable

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

from features.mfcc import extract_mfcc

manifest = pd.read_csv(REPO_ROOT / "data" / "clips" / "manifest.csv")
speakers = sorted(manifest["speaker"].unique())
speakers
```

- [ ] **Step 3: Cell 2 — walk through what MFCC *is* on one clip**

```python
# Slide content: build up MFCC step by step on a single clip.
sample = manifest[manifest["speaker"] == speakers[0]].iloc[0]
y, sr = librosa.load(REPO_ROOT / sample["clip_path"], sr=16000)

fig, axes = plt.subplots(3, 1, figsize=(10, 6))
librosa.display.waveshow(y, sr=sr, ax=axes[0])
axes[0].set_title("1) Raw waveform")

mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=512, hop_length=160)
librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr, x_axis="time", y_axis="mel", ax=axes[1])
axes[1].set_title("2) Mel spectrogram (log dB)")

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=512, hop_length=160, win_length=400)
librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=axes[2])
axes[2].set_title("3) MFCC coefficients (cepstral)")

plt.tight_layout()
```

- [ ] **Step 4: Cell 3 — one MFCC descriptor per speaker, side by side**

```python
fig, axes = plt.subplots(1, len(speakers), figsize=(3 * len(speakers), 4), sharey=True)
if len(speakers) == 1:
    axes = [axes]
for ax, speaker in zip(axes, speakers):
    sample = manifest[manifest["speaker"] == speaker].iloc[0]
    vec = extract_mfcc(str(REPO_ROOT / sample["clip_path"]))
    ax.bar(range(120), vec)
    ax.set_title(speaker)
    ax.set_xlabel("descriptor dim (60 mean + 60 std)")
plt.tight_layout()
plt.savefig(REPO_ROOT / "docs" / "figures" / "mfcc_by_speaker.png", dpi=120, bbox_inches="tight")
plt.show()
```

- [ ] **Step 5: Write `notebooks/03_features_melspec.ipynb` — Cell 1**

```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

from features.spectrogram import extract_logmel

manifest = pd.read_csv(REPO_ROOT / "data" / "clips" / "manifest.csv")
speakers = sorted(manifest["speaker"].unique())
```

- [ ] **Step 6: Cell 2 — one log-mel per speaker, side by side**

```python
fig, axes = plt.subplots(1, len(speakers), figsize=(4 * len(speakers), 4), sharey=True)
if len(speakers) == 1:
    axes = [axes]
for ax, speaker in zip(axes, speakers):
    sample = manifest[manifest["speaker"] == speaker].iloc[0]
    spec = extract_logmel(str(REPO_ROOT / sample["clip_path"]))
    im = librosa.display.specshow(spec, sr=16000, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(speaker)
fig.colorbar(im, ax=axes, format="%+2.0f")
plt.savefig(REPO_ROOT / "docs" / "figures" / "melspec_by_speaker.png", dpi=120, bbox_inches="tight")
plt.show()
```

- [ ] **Step 7: Smoke test both notebooks**

```bash
source venv/bin/activate
cd notebooks
jupyter nbconvert --to notebook --execute 02_features_mfcc.ipynb --output 02_features_mfcc.ipynb
jupyter nbconvert --to notebook --execute 03_features_melspec.ipynb --output 03_features_melspec.ipynb
ls ../docs/figures/
```
Expected: both notebooks succeed. `mfcc_by_speaker.png` and `melspec_by_speaker.png` exist in `docs/figures/`.

- [ ] **Step 8: Commit**

```bash
git add notebooks/02_features_mfcc.ipynb notebooks/03_features_melspec.ipynb docs/figures/.gitkeep 2>/dev/null || true
touch docs/figures/.gitkeep
git add docs/figures/.gitkeep notebooks/02_features_mfcc.ipynb notebooks/03_features_melspec.ipynb
git commit -m "feat(notebooks): feature-visualization notebooks for MFCC and log-mel"
```

(Note: `docs/figures/*.png` are gitignored — we commit only the `.gitkeep` so the directory exists in fresh clones.)

---

## Task 7: SVM training notebook (`04_train_svm.ipynb`)

**Files:**
- Create: `notebooks/04_train_svm.ipynb`
- Create: `models/artifacts/` (directory, gitignored contents)

**Interfaces:**
- Consumes: `data/clips/manifest.csv`, `data/clips/splits.csv`, `features/mfcc.py`.
- Produces: `models/artifacts/svm.joblib` with keys `{"scaler": StandardScaler, "model": SVC, "label_map": list[str]}`. Consumed by Tasks 9 (eval) and 10 (server).

- [ ] **Step 1: Cell 1 — setup and build feature matrices**

```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from features.mfcc import extract_mfcc

ART_DIR = REPO_ROOT / "models" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(REPO_ROOT / "data" / "clips" / "manifest.csv")
splits = pd.read_csv(REPO_ROOT / "data" / "clips" / "splits.csv")
data = manifest.merge(splits, on="clip_path")

label_map = sorted(data["speaker"].unique())
label_to_idx = {name: i for i, name in enumerate(label_map)}


def build_xy(split_name: str):
    subset = data[data["split"] == split_name]
    xs = np.stack([extract_mfcc(str(REPO_ROOT / p)) for p in tqdm(subset["clip_path"])])
    ys = np.array([label_to_idx[s] for s in subset["speaker"]])
    return xs, ys


X_train, y_train = build_xy("train")
X_val, y_val = build_xy("val")
X_test, y_test = build_xy("test")
print("shapes:", X_train.shape, X_val.shape, X_test.shape)
```

- [ ] **Step 2: Cell 2 — scale, grid search, pick best**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

grid = [(C, g) for C in (1, 10, 100) for g in ("scale", "auto", 0.01)]
best_acc = -1.0
best_cfg = None
best_model = None
for C, g in grid:
    svc = SVC(kernel="rbf", C=C, gamma=g, probability=True, random_state=42)
    svc.fit(X_train_s, y_train)
    acc = svc.score(X_val_s, y_val)
    print(f"C={C:>4}  gamma={g!s:>6}  val_acc={acc:.3f}")
    if acc > best_acc:
        best_acc = acc
        best_cfg = (C, g)
        best_model = svc

print("best:", best_cfg, "val_acc=", best_acc)
```

- [ ] **Step 3: Cell 3 — persist artifact**

```python
import joblib

joblib.dump(
    {"scaler": scaler, "model": best_model, "label_map": label_map},
    ART_DIR / "svm.joblib",
)
print("saved", ART_DIR / "svm.joblib")
```

- [ ] **Step 4: Smoke test**

```bash
source venv/bin/activate
cd notebooks
jupyter nbconvert --to notebook --execute 04_train_svm.ipynb --output 04_train_svm.ipynb
ls -la ../models/artifacts/
```
Expected: notebook runs cleanly. `models/artifacts/svm.joblib` exists. Best val accuracy printed and > 0.5 (well above 0.25 random-4-class).

- [ ] **Step 5: Commit**

```bash
git add notebooks/04_train_svm.ipynb
git commit -m "feat(notebooks): SVM baseline training on MFCC features"
```

---

## Task 8: CNN training notebook (`05_train_cnn.ipynb`)

**Files:**
- Create: `notebooks/05_train_cnn.ipynb`

**Interfaces:**
- Consumes: `data/clips/manifest.csv`, `data/clips/splits.csv`, `features/spectrogram.py`, `models/cnn.py`.
- Produces: `models/artifacts/cnn.pt` with keys `{"state_dict": dict, "label_map": list[str]}`. Consumed by Tasks 9 (eval) and 10 (server).

- [ ] **Step 1: Cell 1 — imports and dataset**

```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from features.spectrogram import extract_logmel
from models.cnn import SpeakerCNN

ART_DIR = REPO_ROOT / "models" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(REPO_ROOT / "data" / "clips" / "manifest.csv")
splits = pd.read_csv(REPO_ROOT / "data" / "clips" / "splits.csv")
data = manifest.merge(splits, on="clip_path")

label_map = sorted(data["speaker"].unique())
label_to_idx = {name: i for i, name in enumerate(label_map)}


class ClipDataset(Dataset):
    def __init__(self, rows: pd.DataFrame):
        self.rows = rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        spec = extract_logmel(str(REPO_ROOT / row["clip_path"]))
        x = torch.from_numpy(spec).unsqueeze(0)  # (1, 64, 300)
        y = label_to_idx[row["speaker"]]
        return x, y


train_ds = ClipDataset(data[data["split"] == "train"])
val_ds = ClipDataset(data[data["split"] == "val"])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
print(f"train={len(train_ds)}  val={len(val_ds)}")
```

- [ ] **Step 2: Cell 2 — training loop with early stopping**

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = SpeakerCNN(num_classes=len(label_map)).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 30
PATIENCE = 5
best_val_loss = float("inf")
patience_left = PATIENCE
history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_state = None

for epoch in range(EPOCHS):
    model.train()
    tl = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        tl += loss.item() * x.size(0)
    tl /= len(train_ds)

    model.eval()
    vl = 0.0
    correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            vl += criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
    vl /= len(val_ds)
    va = correct / len(val_ds)
    history["train_loss"].append(tl)
    history["val_loss"].append(vl)
    history["val_acc"].append(va)
    print(f"epoch {epoch:02d}  train={tl:.3f}  val={vl:.3f}  val_acc={va:.3f}")

    if vl < best_val_loss - 1e-4:
        best_val_loss = vl
        patience_left = PATIENCE
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_left -= 1
        if patience_left <= 0:
            print("early stop")
            break
```

- [ ] **Step 3: Cell 3 — plot curves + persist artifact**

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(history["train_loss"], label="train")
ax[0].plot(history["val_loss"], label="val")
ax[0].set_title("loss")
ax[0].legend()
ax[1].plot(history["val_acc"])
ax[1].set_title("val accuracy")
plt.tight_layout()
plt.savefig(REPO_ROOT / "docs" / "figures" / "cnn_training.png", dpi=120, bbox_inches="tight")
plt.show()

torch.save(
    {"state_dict": best_state, "label_map": label_map},
    ART_DIR / "cnn.pt",
)
print("saved", ART_DIR / "cnn.pt")
```

- [ ] **Step 4: Smoke test**

```bash
source venv/bin/activate
cd notebooks
jupyter nbconvert --to notebook --execute 05_train_cnn.ipynb --output 05_train_cnn.ipynb
ls -la ../models/artifacts/
```
Expected: notebook runs cleanly (a few minutes on GPU, longer on CPU). `models/artifacts/cnn.pt` exists. `docs/figures/cnn_training.png` exists. Final val accuracy printed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/05_train_cnn.ipynb
git commit -m "feat(notebooks): CNN training on log-mel spectrograms"
```

---

## Task 9: Evaluation notebook (`06_evaluate_and_compare.ipynb`)

**Files:**
- Create: `notebooks/06_evaluate_and_compare.ipynb`

**Interfaces:**
- Consumes: `models/artifacts/svm.joblib`, `models/artifacts/cnn.pt`, `data/clips/manifest.csv`, `data/clips/splits.csv`.
- Produces: `docs/figures/confmat_svm.png`, `docs/figures/confmat_cnn.png`, `docs/figures/f1_compare.png`. No new module code.

- [ ] **Step 1: Cell 1 — load everything**

```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt

from features.mfcc import extract_mfcc
from features.spectrogram import extract_logmel
from models.cnn import SpeakerCNN
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay

FIG_DIR = REPO_ROOT / "docs" / "figures"

manifest = pd.read_csv(REPO_ROOT / "data" / "clips" / "manifest.csv")
splits = pd.read_csv(REPO_ROOT / "data" / "clips" / "splits.csv")
data = manifest.merge(splits, on="clip_path")
test = data[data["split"] == "test"].reset_index(drop=True)

svm_art = joblib.load(REPO_ROOT / "models" / "artifacts" / "svm.joblib")
cnn_art = torch.load(REPO_ROOT / "models" / "artifacts" / "cnn.pt", map_location="cpu")
label_map = svm_art["label_map"]
assert cnn_art["label_map"] == label_map, "label_map mismatch between artifacts"
label_to_idx = {n: i for i, n in enumerate(label_map)}

y_true = np.array([label_to_idx[s] for s in test["speaker"]])
print("test clips:", len(test), "classes:", label_map)
```

- [ ] **Step 2: Cell 2 — SVM predictions**

```python
X_test = np.stack([extract_mfcc(str(REPO_ROOT / p)) for p in test["clip_path"]])
X_test_s = svm_art["scaler"].transform(X_test)
y_pred_svm = svm_art["model"].predict(X_test_s)
print("SVM accuracy:", (y_pred_svm == y_true).mean())
print(classification_report(y_true, y_pred_svm, target_names=label_map))
```

- [ ] **Step 3: Cell 3 — CNN predictions**

```python
model = SpeakerCNN(num_classes=len(label_map))
model.load_state_dict(cnn_art["state_dict"])
model.eval()

y_pred_cnn = []
with torch.no_grad():
    for p in test["clip_path"]:
        spec = extract_logmel(str(REPO_ROOT / p))
        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 300)
        y_pred_cnn.append(int(model(x).argmax(dim=1).item()))
y_pred_cnn = np.array(y_pred_cnn)
print("CNN accuracy:", (y_pred_cnn == y_true).mean())
print(classification_report(y_true, y_pred_cnn, target_names=label_map))
```

- [ ] **Step 4: Cell 4 — confusion matrices**

```python
for name, y_pred in [("svm", y_pred_svm), ("cnn", y_pred_cnn)]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_map))))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_map)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name.upper()} — confusion matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"confmat_{name}.png", dpi=120, bbox_inches="tight")
    plt.show()
```

- [ ] **Step 5: Cell 5 — per-class F1 comparison**

```python
f1_svm = f1_score(y_true, y_pred_svm, average=None)
f1_cnn = f1_score(y_true, y_pred_cnn, average=None)

x = np.arange(len(label_map))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - width / 2, f1_svm, width, label="SVM")
ax.bar(x + width / 2, f1_cnn, width, label="CNN")
ax.set_xticks(x, label_map, rotation=20)
ax.set_ylabel("F1")
ax.set_ylim(0, 1)
ax.set_title("Per-class F1: SVM vs CNN")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "f1_compare.png", dpi=120, bbox_inches="tight")
plt.show()
```

- [ ] **Step 6: Cell 6 — robustness (noise + short clip)**

```python
def score_with_noise(y_pred_getter, snr_db: float) -> float:
    """Reuse the same feature helper by writing noisy clip to a temp file."""
    import tempfile, soundfile as sf, librosa
    correct = 0
    for i, row in test.iterrows():
        y, sr = librosa.load(REPO_ROOT / row["clip_path"], sr=16000, mono=True)
        rms = np.sqrt(np.mean(y ** 2)) + 1e-8
        noise_rms = rms / (10 ** (snr_db / 20))
        noisy = y + np.random.randn(len(y)) * noise_rms
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, noisy.astype(np.float32), sr)
            pred = y_pred_getter(f.name)
        if pred == y_true[i]:
            correct += 1
    return correct / len(test)


def svm_pred(path):
    v = extract_mfcc(path).reshape(1, -1)
    return int(svm_art["model"].predict(svm_art["scaler"].transform(v))[0])


def cnn_pred(path):
    x = torch.from_numpy(extract_logmel(path)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return int(model(x).argmax(dim=1).item())


np.random.seed(42)
print("SVM @ 15 dB SNR:", score_with_noise(svm_pred, 15))
print("CNN @ 15 dB SNR:", score_with_noise(cnn_pred, 15))
```

- [ ] **Step 7: Cell 7 — short-clip robustness (1.5 s)**

```python
def score_with_short_crop(y_pred_getter, target_seconds: float = 1.5) -> float:
    """Center-crop test clips to <3s, resave, re-score. Shows value of 3s window."""
    import tempfile, soundfile as sf, librosa
    n_target = int(target_seconds * 16000)
    correct = 0
    for i, row in test.iterrows():
        y, sr = librosa.load(REPO_ROOT / row["clip_path"], sr=16000, mono=True)
        start = (len(y) - n_target) // 2
        cropped = y[start : start + n_target]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, cropped.astype(np.float32), sr)
            pred = y_pred_getter(f.name)
        if pred == y_true[i]:
            correct += 1
    return correct / len(test)


print("SVM @ 1.5s clips:", score_with_short_crop(svm_pred))
print("CNN @ 1.5s clips:", score_with_short_crop(cnn_pred))
```

- [ ] **Step 8: Smoke test**

```bash
source venv/bin/activate
cd notebooks
jupyter nbconvert --to notebook --execute 06_evaluate_and_compare.ipynb --output 06_evaluate_and_compare.ipynb
ls -la ../docs/figures/
```
Expected: notebook runs cleanly. `confmat_svm.png`, `confmat_cnn.png`, `f1_compare.png` exist.

- [ ] **Step 9: Commit**

```bash
git add notebooks/06_evaluate_and_compare.ipynb
git commit -m "feat(notebooks): evaluation and comparison with confusion matrices"
```

---

## Task 10: FastAPI server (`scripts/serve.py`)

**Files:**
- Create: `scripts/serve.py`
- Create: `tests/test_serve.py`

**Interfaces:**
- Consumes: `models/artifacts/svm.joblib`, `models/artifacts/cnn.pt`, both feature helpers, `frontend/templates/index.html`, `frontend/static/**`.
- Produces: HTTP endpoints `GET /`, `GET /static/*`, `POST /predict`. `POST /predict` returns the JSON schema defined in the spec (section 6.1). Consumed by Task 11 (frontend).

The frontend directory does not exist yet; Task 10 uses a placeholder `index.html` so the server tests pass. Task 11 replaces the placeholder with the forked bongoSTEM shell.

- [ ] **Step 1: Create placeholder frontend files so tests can run**

```bash
mkdir -p frontend/templates frontend/static
```

Create `frontend/templates/index.html` (temporary placeholder — replaced in Task 11):
```html
<!doctype html>
<html><head><title>speaker-id</title></head>
<body><h1>speaker-id placeholder</h1></body></html>
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_serve.py`:
```python
import io
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """
    Build a test client with fake artifacts injected into module globals
    so we don't need real trained models on disk to test wiring.
    """
    from scripts import serve

    class FakeSVM:
        def predict_proba(self, X):
            # deterministic: always class 0 with high confidence
            n = X.shape[0]
            probs = np.zeros((n, 4))
            probs[:, 0] = 0.9
            probs[:, 1:] = 0.1 / 3
            return probs

    class FakeScaler:
        def transform(self, X):
            return X

    class FakeCNN:
        def eval(self):
            return self
        def __call__(self, x):
            import torch
            n = x.shape[0]
            logits = torch.zeros(n, 4)
            logits[:, 1] = 5.0  # CNN prefers class 1
            return logits
        def load_state_dict(self, sd):
            pass

    serve.SVM_ART = {"model": FakeSVM(), "scaler": FakeScaler(), "label_map": ["baraka", "standin_a", "standin_b", "standin_c"]}
    serve.CNN_MODEL = FakeCNN()
    serve.LABEL_MAP = ["baraka", "standin_a", "standin_b", "standin_c"]
    return TestClient(serve.app)


def _wav_bytes(duration_s: float = 3.0, sr: int = 16000) -> bytes:
    y = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, int(sr * duration_s), endpoint=False))
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def test_root_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "html" in r.text.lower()


def test_predict_svm_returns_baraka(client):
    r = client.post(
        "/predict",
        files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")},
        data={"model": "svm"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "baraka"
    assert body["model_used"] == "svm"
    assert body["all_scores"]["baraka"] > 0.5
    assert "waveform_png_b64" in body
    assert "spectrogram_png_b64" in body


def test_predict_cnn_returns_standin_a(client):
    r = client.post(
        "/predict",
        files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")},
        data={"model": "cnn"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "standin_a"  # FakeCNN prefers class 1
    assert body["model_used"] == "cnn"


def test_predict_rejects_unknown_model(client):
    r = client.post(
        "/predict",
        files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")},
        data={"model": "bogus"},
    )
    assert r.status_code == 400
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_serve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.serve'` (or similar).

- [ ] **Step 4: Implement `scripts/serve.py`**

Create `scripts/serve.py`:
```python
"""FastAPI live-demo server for speaker-id.

Loads both trained artifacts once at startup. POST /predict accepts an
audio blob + model selector and returns prediction, confidence per class,
plus base64 waveform + spectrogram PNGs.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import joblib
import torch
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # server-side rendering, no GUI
import matplotlib.pyplot as plt
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from features.mfcc import extract_mfcc
from features.spectrogram import extract_logmel
from models.cnn import SpeakerCNN

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND = REPO_ROOT / "frontend"
ART_DIR = REPO_ROOT / "models" / "artifacts"

SVM_ART: dict | None = None
CNN_MODEL: SpeakerCNN | None = None
LABEL_MAP: list[str] | None = None

app = FastAPI(title="speaker-id demo")


def _load_artifacts() -> None:
    """Populate module globals from disk. Skipped in tests (monkey-patched)."""
    global SVM_ART, CNN_MODEL, LABEL_MAP
    svm_path = ART_DIR / "svm.joblib"
    cnn_path = ART_DIR / "cnn.pt"
    if not svm_path.exists() or not cnn_path.exists():
        # Server still boots for tests / dev without trained artifacts.
        return
    SVM_ART = joblib.load(svm_path)
    cnn_state = torch.load(cnn_path, map_location="cpu")
    LABEL_MAP = SVM_ART["label_map"]
    assert cnn_state["label_map"] == LABEL_MAP, "label_map mismatch"
    CNN_MODEL = SpeakerCNN(num_classes=len(LABEL_MAP))
    CNN_MODEL.load_state_dict(cnn_state["state_dict"])
    CNN_MODEL.eval()


_load_artifacts()

# Static assets — mount if directory exists
if (FRONTEND / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    index = FRONTEND / "templates" / "index.html"
    if not index.exists():
        raise HTTPException(500, "frontend/templates/index.html missing")
    return HTMLResponse(index.read_text())


def _save_temp_wav(raw: bytes) -> Path:
    """Decode incoming audio, resample to 16k mono, write a 3-s temp wav."""
    import tempfile
    y, sr = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
    target = 16000 * 3
    if len(y) >= target:
        start = (len(y) - target) // 2
        y = y[start : start + target]
    else:
        y = np.pad(y, (0, target - len(y)))
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(tmp, y.astype(np.float32), 16000)
    return tmp


def _render_png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _waveform_png(wav_path: Path) -> str:
    y, sr = librosa.load(str(wav_path), sr=16000)
    fig, ax = plt.subplots(figsize=(6, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("waveform")
    return _render_png(fig)


def _spec_png(wav_path: Path) -> str:
    spec = extract_logmel(str(wav_path))
    fig, ax = plt.subplots(figsize=(6, 3))
    librosa.display.specshow(spec, sr=16000, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("log-mel spectrogram")
    return _render_png(fig)


def _svm_predict(wav_path: Path) -> tuple[str, dict[str, float]]:
    assert SVM_ART is not None, "SVM artifact not loaded"
    v = extract_mfcc(str(wav_path)).reshape(1, -1)
    v = SVM_ART["scaler"].transform(v)
    probs = SVM_ART["model"].predict_proba(v)[0]
    scores = {name: float(p) for name, p in zip(SVM_ART["label_map"], probs)}
    top = max(scores, key=scores.get)
    return top, scores


def _cnn_predict(wav_path: Path) -> tuple[str, dict[str, float]]:
    assert CNN_MODEL is not None, "CNN artifact not loaded"
    assert LABEL_MAP is not None
    spec = extract_logmel(str(wav_path))
    x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1,1,64,300)
    with torch.no_grad():
        logits = CNN_MODEL(x)
        probs = torch.softmax(logits, dim=1)[0].tolist()
    scores = {name: float(p) for name, p in zip(LABEL_MAP, probs)}
    top = max(scores, key=scores.get)
    return top, scores


@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    model: str = Form(...),
) -> dict:
    if model not in {"svm", "cnn"}:
        raise HTTPException(400, "model must be 'svm' or 'cnn'")
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio blob")

    wav = _save_temp_wav(raw)
    try:
        if model == "svm":
            top, scores = _svm_predict(wav)
        else:
            top, scores = _cnn_predict(wav)
        return {
            "prediction": top,
            "confidence": scores[top],
            "all_scores": scores,
            "waveform_png_b64": _waveform_png(wav),
            "spectrogram_png_b64": _spec_png(wav),
            "model_used": model,
        }
    finally:
        wav.unlink(missing_ok=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_serve.py -v`
Expected: 4 passed.

- [ ] **Step 6: Manual smoke test (if artifacts already trained)**

If `models/artifacts/svm.joblib` and `cnn.pt` exist, start the server:
```bash
source venv/bin/activate
uvicorn scripts.serve:app --reload
```
In a second terminal:
```bash
curl -s http://127.0.0.1:8000/ | head -3
```
Expected: HTML placeholder page comes back. Stop uvicorn with Ctrl+C. Skip this step if artifacts aren't trained yet.

- [ ] **Step 7: Commit**

```bash
git add scripts/serve.py tests/test_serve.py frontend/templates/index.html
git commit -m "feat(serve): FastAPI live-demo server with predict endpoint"
```

---

## Task 11: Frontend fork from bongoSTEM

**Files:**
- Copy into: `frontend/static/css/`, `frontend/static/*.svg` (from `~/bongoSTEM/bongo/frontend/static/`)
- Create: `frontend/static/js/app.js`
- Overwrite: `frontend/templates/index.html`

**Interfaces:**
- Consumes: `POST /predict` (Task 10 shape).
- Produces: interactive record → predict UI. Terminal task of the plan for user-facing functionality.

- [ ] **Step 1: Fork bongoSTEM shell assets**

```bash
cp -r ~/bongoSTEM/bongo/frontend/static/css frontend/static/
cp ~/bongoSTEM/bongo/frontend/static/fulcrum-black-bg-logo.svg frontend/static/
cp ~/bongoSTEM/bongo/frontend/static/fulcrum-transparent-bg.svg frontend/static/
ls frontend/static/
```
Expected: `css/`, both `.svg` files present.

- [ ] **Step 2: Write the new `frontend/templates/index.html`**

Overwrite the placeholder with:
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>speaker-id demo</title>
  <link rel="stylesheet" href="/static/css/style.css" />
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
    header { display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; }
    header img { height: 48px; }
    #panel-top { display: flex; gap: 1rem; align-items: center; margin-bottom: 1.5rem; }
    #record-btn { padding: 1rem 2rem; font-size: 1.2rem; border-radius: 999px; border: 0; cursor: pointer; background: #d33; color: #fff; }
    #record-btn.recording { background: #333; }
    #panel-mid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }
    #panel-mid img { width: 100%; border: 1px solid #eee; border-radius: 4px; }
    #panel-bottom { padding: 1rem; border: 1px solid #eee; border-radius: 6px; }
    #prediction { font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem; }
    #confidence-chart { width: 100%; height: 240px; }
    select, label { font-size: 1rem; }
  </style>
</head>
<body>
  <header>
    <img src="/static/fulcrum-transparent-bg.svg" alt="fulcrum" />
    <h1>speaker-id — live demo</h1>
  </header>

  <div id="panel-top">
    <button id="record-btn">Record</button>
    <label>Model:
      <select id="model-select">
        <option value="svm">SVM (MFCC)</option>
        <option value="cnn">CNN (log-mel)</option>
      </select>
    </label>
    <span id="status"></span>
  </div>

  <div id="panel-mid">
    <img id="waveform" alt="waveform" />
    <img id="spectrogram" alt="spectrogram" />
  </div>

  <div id="panel-bottom">
    <div id="prediction">— press record —</div>
    <div id="confidence-chart"></div>
  </div>

  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="/static/js/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write `frontend/static/js/app.js`**

Create the file:
```javascript
const btn = document.getElementById("record-btn");
const modelSelect = document.getElementById("model-select");
const statusEl = document.getElementById("status");
const wave = document.getElementById("waveform");
const spec = document.getElementById("spectrogram");
const predEl = document.getElementById("prediction");
const chartEl = document.getElementById("confidence-chart");

let mediaRecorder = null;
let chunks = [];
const RECORD_MS = 3500;

async function ensureRecorder() {
  if (mediaRecorder) return mediaRecorder;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.onstop = async () => {
    const blob = new Blob(chunks, { type: "audio/webm" });
    chunks = [];
    await sendClip(blob);
  };
  return mediaRecorder;
}

async function sendClip(blob) {
  statusEl.textContent = "predicting…";
  const fd = new FormData();
  fd.append("audio", blob, "clip.webm");
  fd.append("model", modelSelect.value);
  const res = await fetch("/predict", { method: "POST", body: fd });
  if (!res.ok) {
    statusEl.textContent = `error ${res.status}`;
    return;
  }
  const data = await res.json();
  wave.src = "data:image/png;base64," + data.waveform_png_b64;
  spec.src = "data:image/png;base64," + data.spectrogram_png_b64;
  predEl.textContent = `${data.prediction}  (${(data.confidence * 100).toFixed(1)}% via ${data.model_used.toUpperCase()})`;
  const names = Object.keys(data.all_scores);
  const scores = names.map((n) => data.all_scores[n]);
  Plotly.newPlot(chartEl, [{ x: names, y: scores, type: "bar" }], {
    margin: { t: 20, l: 40, r: 20, b: 40 }, yaxis: { range: [0, 1] },
  }, { displayModeBar: false });
  statusEl.textContent = "";
}

btn.addEventListener("click", async () => {
  const rec = await ensureRecorder();
  if (rec.state === "recording") return;
  btn.classList.add("recording");
  btn.textContent = "Recording…";
  rec.start();
  setTimeout(() => {
    rec.stop();
    btn.classList.remove("recording");
    btn.textContent = "Record";
  }, RECORD_MS);
});
```

- [ ] **Step 4: Manual browser test**

Prerequisite: artifacts trained (Tasks 7 and 8 completed with real data).
```bash
source venv/bin/activate
uvicorn scripts.serve:app --reload
```
Open `http://127.0.0.1:8000/` in Chromium or Firefox. Allow microphone access. Click **Record**, speak for ~3 seconds. Verify:
- Both waveform and spectrogram images appear in the middle panel.
- Prediction text below shows a speaker name and a percentage.
- Confidence bar chart renders with 4 bars.
- Model toggle switches between SVM and CNN and prediction updates on next record.

If Task 10 tests pass but the browser flow errors, check the terminal running uvicorn for the stack trace.

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): fork bongoSTEM shell and wire record -> predict UI"
```

---

## Task 12: Documentation and polish

**Files:**
- Modify: `README.md`
- Create: `docs/design.md`
- Create: `docs/slides.md`
- Modify: `requirements.txt` (pin versions to what's actually installed)

**Interfaces:** none (docs only).

- [ ] **Step 1: Freeze `requirements.txt` to exact installed versions**

```bash
source venv/bin/activate
pip freeze | grep -Ei '^(librosa|scikit-learn|torch|torchaudio|sounddevice|soundfile|fastapi|uvicorn|matplotlib|numpy|scipy|pandas|joblib|jupyter|jupytext|pytest|httpx|python-multipart)==' > requirements.txt
cat requirements.txt
```

- [ ] **Step 2: Rewrite `README.md`**

Overwrite with:
```markdown
# speech-recognition (speaker-id)

Group assignment for the CA Studies **Image Processing → Pattern Recognition** module.
Phase 1: classify a 3-second audio clip as one of 4 speakers using MFCC + SVM
and log-mel + CNN pipelines, with a browser-based live demo.

Full design: [docs/superpowers/specs/2026-07-02-speaker-id-phase1-design.md](docs/superpowers/specs/2026-07-02-speaker-id-phase1-design.md).

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 1. Drop your phone recordings into data/recordings/baraka/
# 2. Run notebooks in order (jupyter notebook, then run each end-to-end):
#    01_ingest.ipynb          -> data/clips/ + manifest + splits
#    02_features_mfcc.ipynb   -> docs/figures/mfcc_by_speaker.png
#    03_features_melspec.ipynb-> docs/figures/melspec_by_speaker.png
#    04_train_svm.ipynb       -> models/artifacts/svm.joblib
#    05_train_cnn.ipynb       -> models/artifacts/cnn.pt
#    06_evaluate_and_compare.ipynb -> docs/figures/confmat_*.png, f1_compare.png

# 3. Start the demo
uvicorn scripts.serve:app --reload
# open http://127.0.0.1:8000/
```

## Tests

```bash
pytest -v
```

## Swapping in real teammate data

When a teammate delivers their phone recordings:

1. Drop their files into `data/recordings/<their_name>/` (e.g. `data/recordings/carol/`).
2. Delete the corresponding placeholder directory (`data/recordings/standin_a/` or whichever).
3. Re-run notebooks 01 through 06 in order.

No code changes — every artifact's `label_map` is derived from directory
names at training time, and the server picks up the new labels on restart.

## Repo layout

See the design spec.
```

- [ ] **Step 3: Create `docs/design.md`**

Create with:
```markdown
# Design

The full Phase 1 design lives at
[../docs/superpowers/specs/2026-07-02-speaker-id-phase1-design.md](superpowers/specs/2026-07-02-speaker-id-phase1-design.md).

## High-level architecture

```
raw audio -> 01_ingest -> clips + manifest + splits
                         |
                         +-> features/mfcc.py    -> 04_train_svm -> svm.joblib
                         |                                       \
                         +-> features/spectrogram.py -> 05_train_cnn -> cnn.pt
                                                                 \       \
                                                                  06_evaluate_and_compare
                                                                        (confusion matrices)

svm.joblib + cnn.pt -> scripts/serve.py (FastAPI) <- frontend/ (record -> predict UI)
```

## Key decisions

- Notebooks orchestrate the ML lifecycle; `.py` files hold reusable helpers only.
- 16 kHz mono, 3-second non-overlapping clips throughout.
- Splits grouped by source file to prevent adjacency leakage.
- LibriSpeech dev-clean speakers as placeholders until teammate data arrives.
```

- [ ] **Step 4: Create `docs/slides.md`**

Create with:
```markdown
# Presentation outline

1. Problem — pattern recognition on audio: given a 3-second clip, name the speaker.
2. Data — how much we recorded, per-speaker minutes, split policy. Show `manifest.csv` counts.
3. Feature extraction
   - MFCC math (mel scale → cepstral coefficients → deltas). Fig: `mfcc_by_speaker.png`.
   - Log-mel spectrogram as an "image." Fig: `melspec_by_speaker.png`.
4. Classical pipeline — SVM (RBF, C, gamma grid). Val curves + best config.
5. Neural pipeline — small CNN on log-mel. Loss + val-accuracy curves (`cnn_training.png`).
6. Results — confusion matrices (`confmat_svm.png`, `confmat_cnn.png`), per-class F1 comparison (`f1_compare.png`).
7. Live demo — record → prediction.
8. Failure cases + noise robustness (SNR 15 dB).
9. Future work — real teammate data swap-in, potential bongoSTEM integration (biometric identity gate).
```

- [ ] **Step 5: Run all tests one last time**

```bash
source venv/bin/activate
pytest -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/design.md docs/slides.md requirements.txt
git commit -m "docs: quickstart, high-level design, PPT outline; freeze deps"
```

---

## Summary of deliverables

After Task 12:
- Working end-to-end pipeline (ingest → features → SVM + CNN → eval).
- Live demo running at `http://127.0.0.1:8000/`.
- All figures for the deck saved in `docs/figures/`.
- `pytest -v` green.
- Ready to swap in real teammate audio: `data/recordings/<name>/` → re-run notebooks → done.
