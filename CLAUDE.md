# speaker_id — CA Studies Group Assignment

## What This Project Is

Group assignment for the **Image Processing → Pattern Recognition** module. Sister group is doing "video processing + object detection." Ours is the audio analog: **classify audio clips**.

The academic framing is **classical pattern recognition**, not full ASR. We turn audio into features (MFCC or mel-spectrograms), train a small classifier, evaluate on a held-out split, present the confusion matrix + a live demo.

**Group size:** 4 people. Each records themselves — the recordings ARE the dataset.

**Deliverable:** working prototype + PPT + presentation with live demo.

---

## Scope (Phased)

### Phase 1 — Speaker Identification (MUST SHIP)
- 4 classes = group members
- Task: given a 3-second clip, name who spoke
- Two model pipelines, both presented:
  - **Classical:** MFCC (13-20 coefficients + delta + delta-delta) → SVM / k-NN / Random Forest
  - **Neural:** log-mel-spectrogram (treated as a 2D image) → 2-3 layer CNN → softmax over 4 classes
- Live demo: record button → show waveform + spectrogram side-by-side → predicted speaker + confidence
- PPT: MFCC math (mel scale, cepstral coefficients), spectrogram → CNN pipeline, confusion matrix, live demo

### Phase 2 — Add Command Recognition (OPTIONAL, only if Phase 1 is solid)
- Same feature pipeline, second classifier over a small vocab: up/down/left/right/stop
- Demo becomes: "Baraka said 'up'."
- Uses Google Speech Commands dataset or self-recorded commands

### Phase 3 — DO NOT ATTEMPT
- Real diarization (overlapping speakers, segmentation + labeling). Dissertation-level. Out of scope.

---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Audio I/O | `sounddevice` (record), `soundfile` (read/write) | Simple, cross-platform |
| Feature extraction | `librosa` | MFCC + mel-spectrogram in 5 lines each |
| Classical model | `scikit-learn` (SVM, RandomForest, KNN) | Fast train, easy interpretation |
| Neural model | PyTorch — small CNN | Same stack Baraka uses in stormVoice |
| Frontend | FastAPI + vanilla JS + CSS | Fork the UI shell from stormVoice (logo, layout) |
| Charting | matplotlib for PPT, Plotly.js for live demo | Confusion matrices, waveform, spectrogram |

**Deps (rough):**
```
librosa>=0.10.0
scikit-learn>=1.4.0
torch>=2.1.0
sounddevice>=0.4.6
soundfile>=0.12.0
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
matplotlib>=3.8.0
numpy>=1.26.0
scipy>=1.11.0
```

---

## Directory Structure

```
speaker_id/
├── data/
│   └── recordings/                # per-person raw WAV files
│       ├── alice/
│       ├── bob/
│       ├── carol/
│       └── dave/
├── features/
│   ├── mfcc.py                    # MFCC + deltas
│   └── spectrogram.py             # log-mel-spectrogram
├── models/
│   ├── classical.py               # SVM baseline
│   └── cnn.py                     # small CNN on spectrograms
├── scripts/
│   ├── record.py                  # capture training data per person
│   ├── train.py                   # train + save models
│   ├── evaluate.py                # confusion matrix, per-class metrics
│   └── serve.py                   # FastAPI live demo
├── notebooks/                     # exploration + PPT figures
├── frontend/
│   ├── static/                    # forked UI shell + logo
│   └── templates/
└── docs/
    ├── design.md                  # architecture, decisions
    └── slides.md                  # PPT outline
```

---

## Data Collection Guidelines

- Each group member records ~10 minutes total (mix of reading passages + natural speech)
- Segment into 3-second clips → ~200 clips per person
- 4 people × 200 clips = 800 training samples, plenty for both SVM and small CNN
- Split 80/10/10 train/val/test
- Record in similar conditions (same room, same mic if possible) — but include SOME variability to test robustness

---

## Coding Rules

- 300-line file limit (same as stormVoice discipline)
- One responsibility per file — features/mfcc.py extracts MFCC, nothing else
- Comments explain WHY, not WHAT
- Trust internal function calls, validate only at boundaries (CLI args, uploaded audio files)
- Fail fast on missing recordings — clear error, no silent defaults

---

## Relationship to stormVoice

Baraka's parallel project. This assignment reuses UI polish from stormVoice (logo, CSS shell) but is otherwise independent. Do NOT modify anything under `~/stormVoice/` from this project.

The trained speaker-ID model has natural future integrations into stormVoice:
- **Biometric identity gate** — verify it's Baraka speaking before unlocking private memories
- **Multi-user awareness** — if a friend joins, Bongo knows who is who
- **Swahili STT training data pipeline** — the recording capture code here feeds directly into the planned Swahili whisper fine-tune

---

## PPT Structure (draft)

1. Problem statement — pattern recognition on audio
2. Data — what we recorded, why, how much
3. Feature extraction — MFCC (with math) + mel-spectrogram (the "image processing" tie)
4. Classical pipeline — SVM on MFCC (baseline)
5. Neural pipeline — CNN on mel-spectrogram (the arc from feature engineering to representation learning)
6. Results — confusion matrix, per-class F1, comparison
7. Live demo
8. Failure cases + limitations (mic distance, background noise, similar voices)
9. Future work (this is where you can mention stormVoice integrations)

---

## Environment

- OS: Linux (Ubuntu 22.04)
- Python: 3.10+
- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM) — training on this
- Git: initialize on first commit
- Remote: TBD (github.com/Baraka-Malila/speaker-id probably)
