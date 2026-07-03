# stormVoice Lite — Design

**Date:** 2026-07-03  
**Owner:** Baraka Malila (project lead, delivering base version)  
**Approved approach:** Approach A — full stormVoice Lite

---

## What it does

Given a recorded voice clip, stormVoice Lite:
1. Identifies who is speaking (from enrolled group members)
2. Transcribes what they said (faster-whisper)
3. Detects fraud-related language and returns a risk score, category, and recommended action

Output per analysis: speaker name, confidence, transcript, detected signals, fraud category, risk score, risk level, recommended action.

---

## Tech stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Backend | FastAPI + uvicorn | As specified |
| Speaker ID | MFCC (120,) + nearest-centroid | Classical pattern recognition framing; enrollment-compatible |
| STT | faster-whisper (tiny/base model) | Local, offline |
| Fraud engine | Rule-based keyword scoring | Explainable; from PDF section 10 |
| Database | SQLite + SQLAlchemy | Zero-setup; swap to Postgres with one env var change |
| Frontend | Vanilla JS + bongoSTEM CSS fork | Tweaks to bongoSTEM shell as needed |

---

## Speaker identification

**Enrollment (one-time per speaker):**
- Record 3 short clips via browser
- Backend extracts MFCC (120,) per clip → average → store centroid in DB

**Identification:**
- Extract MFCC (120,) from new clip
- Cosine similarity vs. all stored centroids
- Predict speaker with highest similarity above threshold (0.70); else "Unknown"

This is the same as ECAPA-TDNN cosine matching from the PDF but with classical MFCC features — pedagogically appropriate for the Pattern Recognition module.

---

## Fraud engine

Keyword scoring from PDF section 10:

| Keyword/phrase | Points | Category |
|----------------|--------|----------|
| OTP | +40 | OTP theft |
| PIN | +40 | PIN theft |
| Password | +35 | Account takeover |
| Customer care | +20 | Fake authority |
| Account blocked | +25 | Urgency pressure |
| Send money | +30 | Financial manipulation |
| SIM stopped working | +35 | SIM-swap risk |
| Do not tell anyone | +30 | Secrecy pressure |

Risk levels: 0–20 Low, 21–50 Medium, 51–80 High, 81+ Critical.

---

## Database schema (4 tables)

- `speakers` — id, name, created_at
- `speaker_embeddings` — id, speaker_id, embedding_json (120-float array), sample_count
- `voice_sessions` — id, speaker_id, transcript, fraud_category, risk_score, risk_level, recommended_action, created_at
- `fraud_signals` — id, session_id, keyword, points, category

---

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/speakers/enroll | Register a speaker (name + audio clips) |
| GET | /api/speakers | List enrolled speakers |
| POST | /api/analyze | Analyze a recorded clip → full output |
| GET | /api/sessions | Session history |
| GET | /api/sessions/{id} | Single session detail |
| GET | / | Serve frontend |
| GET | /static/* | Static assets |

---

## Frontend (single page, 3 panels)

1. **Top panel:** Record button + speaker toggle (Analyze / Enroll mode) + status
2. **Middle panel:** Waveform PNG + spectrogram PNG (server-rendered, b64)
3. **Bottom panel:** Results dashboard — speaker name + confidence, transcript, detected signals, fraud category, risk badge, recommended action
4. **Sidebar/tab:** Session history list

bongoSTEM CSS shell forked in; tweaks to layout for the dashboard panels.

---

## Repo layout additions (on top of existing structure)

```
speech-recognition/
├── db/
│   ├── __init__.py
│   ├── models.py          # SQLAlchemy ORM models
│   └── session.py         # engine + get_db dependency
├── fraud/
│   ├── __init__.py
│   └── engine.py          # keyword scoring, risk classification
├── stt/
│   ├── __init__.py
│   └── transcribe.py      # faster-whisper wrapper
├── scripts/
│   └── serve.py           # FastAPI app (replaces old serve.py)
└── frontend/
    ├── static/
    │   ├── css/            # forked from bongoSTEM
    │   ├── js/app.js       # vanilla JS
    │   └── *.svg
    └── templates/
        └── index.html
```

Existing `features/` and `models/` directories retained for MFCC + SVM helpers.
