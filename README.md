# stormVoice Lite

Speaker identification and real-time voice fraud detection. Class project for the Speech Processing module.

**What it does:** Record a voice clip → identifies the speaker → transcribes speech → detects fraud language → returns risk level + recommended action.

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn scripts.serve:app --reload
# open http://127.0.0.1:8000
```

First run downloads the faster-whisper `base.en` model (~145 MB, cached after that).

## Usage

1. **Enroll tab** — record 1–3 clips per group member and submit. Each enrolled speaker becomes a known identity.
2. **Analyze tab** — click Record & Analyze, speak for ~4 seconds. You get: speaker name + confidence, transcript, detected fraud signals, risk level, and recommended action.
3. **History tab** — browse all past analysis sessions.

## Swapping in real teammate data

Enrollment is live — teammates open the browser, go to **Enroll**, record clips, submit. No code changes, no retraining needed.

## Stack

| Layer | Choice |
|-------|--------|
| Backend | FastAPI + uvicorn |
| Speaker ID | MFCC (120-dim) nearest-centroid matching |
| STT | faster-whisper base.en |
| Fraud engine | Rule-based keyword scoring |
| Database | SQLite + SQLAlchemy |
| Frontend | Vanilla JS + bongoSTEM CSS shell |

## Tests

```bash
pytest -v
```

## Swap to PostgreSQL

```bash
export DATABASE_URL=postgresql://user:pass@host/dbname
uvicorn scripts.serve:app --reload
```

No other changes needed — SQLAlchemy handles it.
