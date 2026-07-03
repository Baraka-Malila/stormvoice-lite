# Speaker ID — Phase 1 Design

**Date:** 2026-07-02
**Owner:** Baraka Malila
**Scope:** Phase 1 of the CA Studies group assignment (Image Processing → Pattern Recognition). Classify a 3-second audio clip as one of 4 speakers, using both a classical (MFCC + SVM) and a neural (log-mel-spectrogram + small CNN) pipeline, with a browser-based live demo.

Phase 2 (command recognition) and Phase 3 (diarization) are out of scope for this document.

---

## 1. Context & constraints

The group assignment is the audio analog of a sister group's "video processing + object detection" project. Framing is **classical pattern recognition**, not full ASR: raw audio → features → classifier → confusion matrix → live demo.

Constraints shaping the design:

- **4-person group, but only Baraka's voice is available today.** Teammates have not delivered recordings and there is no timeline yet. The design must let the entire pipeline (features, both models, evaluation, live demo) run end-to-end using **placeholder audio** so development does not stall. Placeholder swaps out for real teammate data when it arrives, with zero code changes.
- **Teammates will record on their phones** (voice memo / WhatsApp voice note) and send the files. Baraka records on his phone too — same collection method for everyone to prevent the model from learning "phone vs laptop mic" instead of voices.
- **Notebooks are the primary workflow** for exploration, feature study, training, and evaluation. Rationale: each cell maps to a specific pedagogical step, and the notebooks double as slide source material. The 300-line file rule from `CLAUDE.md` is relaxed for notebooks; it still applies to `.py` files.
- **Reused shell:** the frontend forks the stormVoice static assets (logo, CSS, layout structure). No code in `~/stormVoice/` is modified.

## 2. Data path

### 2.1 Sources

- `data/recordings/baraka/` — Baraka's phone recordings, raw as delivered (any codec ffmpeg reads).
- `data/recordings/standin_a/`, `_b/`, `_c/` — three Mozilla Common Voice speakers chosen to be **acoustically distinct from each other and from Baraka** (e.g., one female English, one male accented-English, one older speaker). ~10 minutes per speaker.

Common Voice is used only as a development placeholder. The submitted deliverable and final live demo will use real teammate recordings.

### 2.2 Collection recipe (per speaker)

- **~10 minutes of raw audio total**, one or two sittings.
- **Content mix:** ~7 minutes reading (book paragraphs, news articles — gives phonetic coverage) + ~3 minutes unscripted speech (describe your day, explain a topic). Reading alone makes the model brittle; some spontaneous speech improves generalization.
- **Environment:** one quiet room, phone held ~30 cm from face, no music, no other voices.
- **Format:** whatever the phone produces. `.m4a`, `.mp3`, `.opus`, `.wav` — the ingest step normalizes.

### 2.3 Ingest pipeline (`notebooks/01_ingest.ipynb`)

1. Walk `data/recordings/<speaker>/*` for every audio file.
2. Decode + resample to **16 kHz mono** via `librosa.load(sr=16000, mono=True)`.
3. Trim leading/trailing silence (`librosa.effects.trim`), then apply an energy-threshold VAD to remove mid-clip silences longer than ~0.5 s.
4. Chop into **non-overlapping 3.0-second windows**. Drop trailing fragments shorter than 3 s.
5. Save each window as `data/clips/<speaker>/<speaker>_<idx>.wav`.
6. Write `data/clips/manifest.csv` with columns `clip_path, speaker, source_file, duration`. Every downstream step reads the manifest, never the filesystem directly.

### 2.4 Split strategy

- **80 / 10 / 10 train / val / test**, stratified by speaker so class proportions match across splits.
- **Grouped by source file:** all clips derived from the same original recording go to the same split. Adjacent 3-s clips from one recording are near-duplicates; if half go to train and half to test, test accuracy is falsely inflated.
- Split is deterministic (fixed seed) and written to `data/clips/splits.csv` (`clip_path, split`).

### 2.5 Sample-count targets

~200 clips × 4 speakers = ~800 total. Sufficient for SVM and comfortable for a small CNN without augmentation.

## 3. Feature engineering

Both feature paths consume the manifest + splits from Section 2. Feature extraction is **lazy per-clip** during training — with 800 clips at ~5 ms each, caching is not worth the complexity. If a training loop later becomes I/O-bound, adding a `.npz` cache is a one-line change with no API impact.

### 3.1 MFCC (SVM path) — `features/mfcc.py`

Public API: one function.

```python
def extract_mfcc(clip_path: str) -> np.ndarray: ...  # shape (120,)
```

Implementation:

- `librosa.feature.mfcc(y, sr=16000, n_mfcc=20, n_fft=512, hop_length=160, win_length=400)` → 25 ms window, 10 ms hop (standard speech settings). Shape `(20, T)` with `T ≈ 300` for 3 s.
- Compute delta and delta-delta with `librosa.feature.delta`. Stack → `(60, T)`.
- Collapse the time axis to a **fixed-length descriptor**: mean + std across time → `(120,)`. SVM needs a fixed-size input, not a sequence.

Notebook `02_features_mfcc.ipynb` walks through the same steps with visualizations of waveform → mel scale → cepstrum → the final vector. This notebook is the source for the MFCC-math slide.

### 3.2 Log-mel-spectrogram (CNN path) — `features/spectrogram.py`

Public API:

```python
def extract_logmel(clip_path: str) -> np.ndarray: ...  # shape (64, 300)
```

Implementation:

- `librosa.feature.melspectrogram(y, sr=16000, n_mels=64, n_fft=512, hop_length=160)` → `(64, ~300)`.
- Log-scale via `librosa.power_to_db(mel, ref=np.max)`.
- Per-clip normalization (subtract mean, divide by std).
- Return as `(64 mel bins, 300 time frames)` — the CNN input.

Notebook `03_features_melspec.ipynb` renders one log-mel per speaker side-by-side — the "audio as image" tie-in for the deck.

## 4. Models

Both models load the manifest + splits, both write their artifact to `models/artifacts/`.

### 4.1 SVM baseline — `notebooks/04_train_svm.ipynb`

- Filter manifest to train split → call `extract_mfcc` per clip → build `X_train (N, 120)`, `y_train (N,)`. Same for val and test.
- `sklearn.preprocessing.StandardScaler` fit on train, applied to val/test.
- `sklearn.svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True)`. RBF because MFCC space is not linearly separable; `probability=True` so the demo can show confidence.
- Small grid search: `C ∈ {1, 10, 100}`, `gamma ∈ {'scale', 'auto', 0.01}`. Pick best by val accuracy. Notebook-sized, no framework needed.
- Persist: `joblib.dump({'scaler': scaler, 'model': svc, 'label_map': [...]}, 'models/artifacts/svm.joblib')`.

### 4.2 Small CNN — `models/cnn.py` + `notebooks/05_train_cnn.ipynb`

`models/cnn.py` holds the architecture only (importable, testable):

```
Input (1, 64, 300)
Conv2d(1→16, 3x3) → BN → ReLU → MaxPool(2x2)
Conv2d(16→32, 3x3) → BN → ReLU → MaxPool(2x2)
Conv2d(32→64, 3x3) → BN → ReLU → AdaptiveAvgPool(1x1)
Flatten → Dropout(0.3) → Linear(64→4)
```

Training loop lives in the notebook:

- Custom `torch.utils.data.Dataset` reads manifest, calls `extract_logmel`, returns `(tensor, label)`.
- `DataLoader(batch_size=32, shuffle=True, num_workers=2)`.
- `Adam(lr=1e-3)`, `CrossEntropyLoss`, 30 epochs, early stop on val loss with patience=5.
- Plot loss + val accuracy curves (slide material).
- Persist: `torch.save({'state_dict': model.state_dict(), 'label_map': [...]}, 'models/artifacts/cnn.pt')`.

### 4.3 Deliberately excluded (Phase 1)

- **Data augmentation** (SpecAugment, additive noise): only if val accuracy stalls. Adds complexity for uncertain gain.
- **Pretrained backbones** (wav2vec, HuBERT): would obscure the pattern-recognition story the assignment asks for.
- **Hyperparameter frameworks** (Optuna, Ray Tune): overkill for 4 classes / 800 clips.

## 5. Evaluation

One notebook — `06_evaluate_and_compare.ipynb` — is the single source of truth for every number and figure that ends up in the deck. Both artifacts are loaded, both scored on the **same held-out test split**.

### 5.1 Metrics per model

- Overall test accuracy.
- **Confusion matrix** (4×4), rendered with matplotlib → saved to `docs/figures/confmat_svm.png` and `confmat_cnn.png`.
- Per-class precision / recall / F1 via `sklearn.metrics.classification_report`.
- Top-3 confused pairs listed in text — feeds the "failure cases" slide.

### 5.2 Comparison figures

- Side-by-side bar chart of per-class F1 (SVM vs CNN) → `docs/figures/f1_compare.png`.
- One-row-per-model summary table for a slide screenshot.

### 5.3 Robustness sanity checks

Cheap, credible slide content:

- Score both models on test clips with **additive white noise** at ~15 dB SNR; report accuracy delta.
- Score on **1.5-second cropped test clips**; shows the value of the 3-second window.

### 5.4 Deliberately excluded

- Cross-validation. Grouped-by-source test split is honest enough for 800 clips.
- Statistical significance testing between SVM and CNN. Wrong scale of assignment.

## 6. Serving (live demo)

Minimal FastAPI app; frontend forks the stormVoice shell for visual identity but exposes one page.

### 6.1 Backend — `scripts/serve.py`

Single file, well under 300 lines. Both artifacts loaded once at startup into module globals.

Endpoints:

- `GET /` → serves `frontend/templates/index.html`.
- `GET /static/*` → forked shell assets (CSS, logo, JS).
- `POST /predict` → `multipart/form-data` with an audio blob from the browser recorder plus a `model` form field (`"svm"` or `"cnn"`). Returns:

  ```json
  {
    "prediction": "baraka",
    "confidence": 0.87,
    "all_scores": {"baraka": 0.87, "standin_a": 0.08, "standin_b": 0.03, "standin_c": 0.02},
    "waveform_png_b64": "...",
    "spectrogram_png_b64": "...",
    "model_used": "svm"
  }
  ```

The two b64 PNGs are rendered server-side with matplotlib on the exact 3-second clip the model saw. Preprocessing at serve time reuses the same `features/` helpers as training — one code path.

Longer-than-3-s input: take the middle 3 s. Shorter: zero-pad. No silence trimming server-side; users will notice a live demo that eats the start of their words.

### 6.2 Frontend — `frontend/static/` + `frontend/templates/index.html`

Forked from `~/stormVoice/bongo/frontend/static/`: copy `css/`, logo SVGs, and the base layout structure from `index.html`. Strip stormVoice-specific features (chess, calendar, sessions).

Single page, three panels:

1. **Top:** large record button (browser `MediaRecorder` API, ~4 s capture) + SVM/CNN toggle.
2. **Middle:** waveform + spectrogram side-by-side (two `<img>` tags fed the b64 PNGs).
3. **Bottom:** predicted speaker (large text) + confidence bar chart across all 4 classes (Plotly.js).

No auth, no session history, no persistence. Pure demo.

### 6.3 Deliberately excluded

Websocket streaming, chunked real-time inference, Docker, per-request model reloading.

## 7. Repo layout

```
speech-recognition/
├── data/
│   ├── recordings/                # raw inputs (gitignored)
│   │   ├── baraka/
│   │   ├── standin_a/
│   │   ├── standin_b/
│   │   └── standin_c/
│   └── clips/                     # 3s normalized clips (gitignored)
│       ├── <speaker>/
│       ├── manifest.csv
│       └── splits.csv
├── features/
│   ├── __init__.py
│   ├── mfcc.py                    # extract_mfcc()
│   └── spectrogram.py             # extract_logmel()
├── models/
│   ├── __init__.py
│   ├── cnn.py                     # nn.Module only
│   └── artifacts/                 # trained models (gitignored)
├── notebooks/
│   ├── 01_ingest.ipynb
│   ├── 02_features_mfcc.ipynb
│   ├── 03_features_melspec.ipynb
│   ├── 04_train_svm.ipynb
│   ├── 05_train_cnn.ipynb
│   └── 06_evaluate_and_compare.ipynb
├── scripts/
│   └── serve.py                   # FastAPI demo
├── frontend/
│   ├── static/                    # forked from stormVoice
│   └── templates/
│       └── index.html
├── docs/
│   ├── design.md                  # high-level architecture, links here
│   ├── slides.md                  # PPT outline
│   ├── figures/                   # confmat_*.png, f1_compare.png
│   └── superpowers/specs/
│       └── 2026-07-02-speaker-id-phase1-design.md
├── requirements.txt
├── CLAUDE.md
└── README.md
```

**Gitignore additions:** `data/recordings/**`, `data/clips/**`, `models/artifacts/**`, `.ipynb_checkpoints/`, `venv/`.

## 8. Workflow

Order to run things end-to-end:

1. Create venv, install `requirements.txt`.
2. Drop phone recordings into `data/recordings/baraka/`.
3. Download three Common Voice speakers into `data/recordings/standin_*/` (helper in `01_ingest.ipynb`).
4. Run `01_ingest.ipynb` — produces clips, manifest, splits.
5. Run `02` and `03` — feature-visualization notebooks (also slide source).
6. Run `04_train_svm.ipynb` — produces `svm.joblib`.
7. Run `05_train_cnn.ipynb` — produces `cnn.pt`.
8. Run `06_evaluate_and_compare.ipynb` — produces confusion matrices and comparison figures.
9. `uvicorn scripts.serve:app --reload` → open `http://localhost:8000`.

## 9. Teammate swap-in procedure

When any teammate delivers audio:

1. Drop their files into `data/recordings/<their_name>/`.
2. Delete the corresponding `standin_*/` directory (or all three, once all teammates deliver).
3. Re-run `01_ingest.ipynb` through `06_evaluate_and_compare.ipynb`.

No code changes. The `label_map` inside each artifact is derived from directory names at training time.

## 10. Success criteria for Phase 1

- Both notebooks 04 and 05 run to completion and produce a valid saved artifact.
- Notebook 06 reports test accuracy > 0.80 for at least one model on the placeholder dataset. (Baseline is 0.25 for random 4-class.)
- `uvicorn scripts.serve:app` starts cleanly, `POST /predict` returns a valid response on a real recorded clip, the frontend renders waveform + spectrogram + confidence bars without errors.
- Confusion matrix PNGs exist in `docs/figures/` and are legible enough to drop into slides.

## 11. Known risks

- **Placeholder mismatch:** Common Voice speakers are typically studio-quality and English-native. Baraka's phone recording will differ acoustically. The classifier may look easier than it is; final numbers on real teammate audio may be worse. Mitigated by picking placeholders that are *acoustically diverse* (accents, ages, mics) and by re-running eval as soon as real data lands.
- **Domain drift at demo time:** the browser microphone in the demo laptop is not the same as the recording phones. If accuracy drops noticeably during the live demo, we plan to add a "training-time noise + reverb" augmentation pass before the presentation.
- **Class imbalance:** if any speaker delivers only 3 minutes instead of 10, that class will be under-represented. The ingest notebook prints per-speaker clip counts prominently so this is visible before training.
