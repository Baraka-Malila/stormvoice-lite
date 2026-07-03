# stormVoice Lite

Speaker identification + real-time fraud detection. Group assignment — Pattern Recognition module.

**What it does:** Record a 4-second voice clip → identify speaker (SVM + CNN) → transcribe speech → detect fraud keywords → return risk level + recommended action.

## Quick start (teammates)

```bash
git clone <repo-url>
cd speech-recognition
bash setup.sh
```

`setup.sh` creates the venv, installs deps, and generates placeholder audio for 4 speakers.

## Train the models

Open each notebook in Jupyter and run all cells:

```bash
source venv/bin/activate
jupyter notebook
```

1. `notebooks/01_classical_pipeline.ipynb` — MFCC → SVM
2. `notebooks/02_neural_pipeline.ipynb`    — Log-mel spectrogram → CNN

Both save trained models to `models/`. Restart the server to pick them up.

## Add your own recordings

```
data/recordings/
  your_name/
    001.wav
    002.wav
    ...    ← 3-second WAV clips, 16 kHz mono
```

Then re-run both notebooks and restart the server.

## Start the server

```bash
uvicorn scripts.serve:app --reload
# open http://127.0.0.1:8000
```

## Architecture

| Layer | Choice |
|---|---|
| Features | MFCC 120-dim (20 coeff + Δ + ΔΔ) / Log-mel 64×300 |
| Classical | SVM (RBF, GridSearchCV) via scikit-learn |
| Neural | 3-layer CNN via PyTorch |
| STT | faster-whisper base.en |
| Fraud | Rule-based keyword scoring |
| Backend | FastAPI + SQLite/SQLAlchemy |
| Frontend | Vanilla JS, bongoSTEM CSS shell |

## Tests

```bash
pytest -v
```

## Swap to PostgreSQL

```bash
export DATABASE_URL=postgresql://user:pass@host/dbname
uvicorn scripts.serve:app --reload
```
