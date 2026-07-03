# stormVoice Lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build stormVoice Lite end-to-end — speaker enrollment + identification (MFCC nearest-centroid), faster-whisper STT, rule-based fraud engine, SQLite session history, FastAPI backend, vanilla JS dashboard forked from stormVoice.

**Architecture:** Seven Jupyter-free Python modules handle distinct concerns (features, DB, speaker ID, fraud, STT, server). FastAPI wires them together and serves the vanilla JS single-page frontend. SQLite + SQLAlchemy stores speakers, embeddings, sessions, and fraud signals. The entire stack runs with `uvicorn scripts.serve:app --reload` after installing deps.

**Tech Stack:** Python 3.10+, librosa, scikit-learn, faster-whisper, SQLAlchemy, FastAPI + uvicorn, matplotlib, soundfile, pytest, httpx, python-multipart, aiofiles.

## Global Constraints

- **Python 3.10+** — all type hints use `X | Y` syntax.
- **300-line limit** on `.py` files (from CLAUDE.md).
- **One responsibility per `.py` file.**
- **16 kHz mono WAV** is the canonical format at every processing boundary.
- **Do NOT modify anything under `~/stormVoice/`** — copy assets only.
- **DATABASE_URL** is read from the environment; defaults to `sqlite:///./stormvoice.db`.
- **Frequent commits** — one per finished task minimum.
- **Repo root:** `/home/cyberpunk/speech-recognition/`
- **Venv path:** `venv/` in repo root. All commands assume `source venv/bin/activate` already run.
- **MFCC similarity threshold:** 0.70. Below this, speaker is "Unknown".
- **faster-whisper model:** `base.en` (downloads ~145 MB on first use; no GPU required, `compute_type="int8"`).

## File Structure

**Files created in this plan:**

```
features/mfcc.py              extract_mfcc() → (120,) float32
features/spectrogram.py       extract_logmel() → (64, 300) float32  [viz only]
db/__init__.py
db/models.py                  SQLAlchemy ORM: Speaker, SpeakerEmbedding, VoiceSession, FraudSignal
db/session.py                 engine, SessionLocal, init_db(), get_db()
speaker_id/__init__.py
speaker_id/identify.py        enroll_speaker(), identify_speaker()
fraud/__init__.py
fraud/engine.py               analyze_fraud() → FraudResult
stt/__init__.py
stt/transcribe.py             transcribe() → str
scripts/serve.py              FastAPI app — all 7 endpoints
frontend/templates/index.html single-page UI
frontend/static/js/app.js     vanilla JS — record, enroll, analyze, history
frontend/static/css/          copied from ~/stormVoice/bongo/frontend/static/css/ (style.css only)
frontend/static/fulcrum-transparent-bg.svg  copied from stormVoice
tests/test_mfcc.py
tests/test_spectrogram.py
tests/test_fraud.py
tests/test_identify.py
tests/test_serve.py
```

**Files modified:**

```
requirements.txt              add sqlalchemy, faster-whisper, aiofiles
.gitignore                    add stormvoice.db, data/samples/
README.md                     replace with quickstart
```

---

## Task 1: Deps, scaffolding, directory skeleton

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `db/__init__.py`, `speaker_id/__init__.py`, `fraud/__init__.py`, `stt/__init__.py`
- Create: `frontend/templates/`, `frontend/static/js/`, `data/samples/`

**Interfaces:**
- Produces: a working venv with all deps; empty package skeletons importable.

- [ ] **Step 1: Overwrite `requirements.txt`**

```
librosa>=0.10.0
scikit-learn>=1.4.0
soundfile>=0.12.0
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
matplotlib>=3.8.0
numpy>=1.26.0
scipy>=1.11.0
pandas>=2.1.0
joblib>=1.3.0
pytest>=7.4.0
httpx>=0.26.0
python-multipart>=0.0.9
sqlalchemy>=2.0.0
faster-whisper>=1.0.0
aiofiles>=23.2.1
```

- [ ] **Step 2: Install deps**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Expected: no `ResolutionImpossible` errors. `faster_whisper`, `sqlalchemy`, `aiofiles` importable.

- [ ] **Step 3: Append to `.gitignore`**

Add these lines to the existing `.gitignore`:

```
stormvoice.db
data/samples/
```

- [ ] **Step 4: Create empty `__init__.py` files**

```bash
touch db/__init__.py speaker_id/__init__.py fraud/__init__.py stt/__init__.py
```

- [ ] **Step 5: Create directory skeleton**

```bash
mkdir -p frontend/templates frontend/static/js frontend/static/css data/samples
```

- [ ] **Step 6: Verify imports**

```bash
python -c "import librosa, sqlalchemy, faster_whisper, fastapi, aiofiles; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore db/__init__.py speaker_id/__init__.py fraud/__init__.py stt/__init__.py frontend/ data/
git commit -m "chore: add deps, scaffolding for stormVoice Lite"
```

---

## Task 2: Feature helpers

**Files:**
- Create: `features/mfcc.py`
- Create: `features/spectrogram.py`
- Create: `tests/test_mfcc.py`
- Create: `tests/test_spectrogram.py`

**Interfaces:**
- Produces:
  - `extract_mfcc(clip_path: str) -> np.ndarray` — shape `(120,)` dtype `float32`. Consumed by Tasks 3 and 6.
  - `extract_logmel(clip_path: str) -> np.ndarray` — shape `(64, 300)` dtype `float32`. Consumed by Task 6 (visualization only).

- [ ] **Step 1: Write `tests/test_mfcc.py`**

```python
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path
from features.mfcc import extract_mfcc


@pytest.fixture
def sine_wav(tmp_path: Path) -> Path:
    sr = 16000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    p = tmp_path / "sine.wav"
    sf.write(p, y, sr)
    return p


def test_shape(sine_wav):
    vec = extract_mfcc(str(sine_wav))
    assert vec.shape == (120,)
    assert vec.dtype == np.float32


def test_deterministic(sine_wav):
    assert np.array_equal(extract_mfcc(str(sine_wav)), extract_mfcc(str(sine_wav)))


def test_short_clip(tmp_path):
    sr = 16000
    y = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, int(sr * 1.5), endpoint=False))).astype(np.float32)
    p = tmp_path / "short.wav"
    sf.write(p, y, sr)
    assert extract_mfcc(str(p)).shape == (120,)
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_mfcc.py -v
```

Expected: `ModuleNotFoundError: No module named 'features.mfcc'`

- [ ] **Step 3: Write `features/mfcc.py`**

```python
from __future__ import annotations
import numpy as np
import librosa

_SR = 16000
_N_MFCC = 20
_N_FFT = 512
_HOP = 160   # 10 ms
_WIN = 400   # 25 ms


def extract_mfcc(clip_path: str) -> np.ndarray:
    y, _ = librosa.load(clip_path, sr=_SR, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=_SR, n_mfcc=_N_MFCC,
                                  n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack([mfcc, d1, d2])          # (60, T)
    descriptor = np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)])  # (120,)
    return descriptor.astype(np.float32)
```

- [ ] **Step 4: Run — expect 3 passed**

```bash
pytest tests/test_mfcc.py -v
```

- [ ] **Step 5: Write `tests/test_spectrogram.py`**

```python
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path
from features.spectrogram import extract_logmel


@pytest.fixture
def sine_3s(tmp_path):
    sr = 16000
    y = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3.0, sr * 3, endpoint=False))).astype(np.float32)
    p = tmp_path / "sine.wav"
    sf.write(p, y, sr)
    return p


def test_shape(sine_3s):
    s = extract_logmel(str(sine_3s))
    assert s.shape == (64, 300)
    assert s.dtype == np.float32


def test_normalized(sine_3s):
    s = extract_logmel(str(sine_3s))
    assert abs(s.mean()) < 1e-3
    assert abs(s.std() - 1.0) < 1e-3


def test_pads_short(tmp_path):
    sr = 16000
    y = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, int(sr * 1.5), endpoint=False))).astype(np.float32)
    p = tmp_path / "short.wav"
    sf.write(p, y, sr)
    assert extract_logmel(str(p)).shape == (64, 300)


def test_crops_long(tmp_path):
    sr = 16000
    y = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 5.0, sr * 5, endpoint=False))).astype(np.float32)
    p = tmp_path / "long.wav"
    sf.write(p, y, sr)
    assert extract_logmel(str(p)).shape == (64, 300)
```

- [ ] **Step 6: Run — expect ImportError**

```bash
pytest tests/test_spectrogram.py -v
```

- [ ] **Step 7: Write `features/spectrogram.py`**

```python
from __future__ import annotations
import numpy as np
import librosa

_SR = 16000
_N_MELS = 64
_N_FFT = 512
_HOP = 160
_TARGET_FRAMES = 300
_TARGET_SAMPLES = _SR * 3


def _fit_to_3s(y: np.ndarray) -> np.ndarray:
    n = len(y)
    if n == _TARGET_SAMPLES:
        return y
    if n > _TARGET_SAMPLES:
        start = (n - _TARGET_SAMPLES) // 2
        return y[start:start + _TARGET_SAMPLES]
    out = np.zeros(_TARGET_SAMPLES, dtype=y.dtype)
    out[:n] = y
    return out


def extract_logmel(clip_path: str) -> np.ndarray:
    y, _ = librosa.load(clip_path, sr=_SR, mono=True)
    y = _fit_to_3s(y)
    mel = librosa.feature.melspectrogram(y=y, sr=_SR, n_mels=_N_MELS,
                                          n_fft=_N_FFT, hop_length=_HOP)
    log_mel = librosa.power_to_db(mel, ref=np.max)[:, :_TARGET_FRAMES]
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)
```

- [ ] **Step 8: Run — expect 4 passed**

```bash
pytest tests/test_spectrogram.py -v
```

- [ ] **Step 9: Commit**

```bash
git add features/mfcc.py features/spectrogram.py tests/test_mfcc.py tests/test_spectrogram.py
git commit -m "feat(features): MFCC (120,) and log-mel (64x300) helpers"
```

---

## Task 3: Database models + speaker identification

**Files:**
- Create: `db/models.py`
- Create: `db/session.py`
- Create: `speaker_id/identify.py`
- Create: `tests/test_identify.py`

**Interfaces:**
- Consumes: `extract_mfcc(clip_path)` from `features.mfcc`.
- Produces:
  - `init_db()` — creates all tables. Called once at server startup.
  - `get_db()` — FastAPI dependency yielding a `Session`.
  - `enroll_speaker(name: str, clip_paths: list[str], db: Session) -> Speaker`
  - `identify_speaker(clip_path: str, db: Session) -> tuple[str, float]` — returns `(name, confidence)` or `("Unknown", similarity_float)`.

- [ ] **Step 1: Write `db/models.py`**

```python
from __future__ import annotations
import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Speaker(Base):
    __tablename__ = "speakers"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    embedding = relationship("SpeakerEmbedding", back_populates="speaker", uselist=False,
                             cascade="all, delete-orphan")
    sessions = relationship("VoiceSession", back_populates="speaker")


class SpeakerEmbedding(Base):
    __tablename__ = "speaker_embeddings"
    id = Column(Integer, primary_key=True)
    speaker_id = Column(Integer, ForeignKey("speakers.id"), nullable=False)
    embedding_json = Column(Text, nullable=False)
    sample_count = Column(Integer, default=0)
    speaker = relationship("Speaker", back_populates="embedding")

    def get_vector(self) -> list[float]:
        return json.loads(self.embedding_json)

    def set_vector(self, vec: list[float]) -> None:
        self.embedding_json = json.dumps([float(v) for v in vec])


class VoiceSession(Base):
    __tablename__ = "voice_sessions"
    id = Column(Integer, primary_key=True)
    speaker_id = Column(Integer, ForeignKey("speakers.id"), nullable=True)
    transcript = Column(Text, default="")
    fraud_category = Column(String, default="Normal Request")
    risk_score = Column(Integer, default=0)
    risk_level = Column(String, default="Low")
    recommended_action = Column(String, default="Allow and log.")
    created_at = Column(DateTime, default=datetime.utcnow)
    speaker = relationship("Speaker", back_populates="sessions")
    fraud_signals = relationship("FraudSignal", back_populates="session",
                                  cascade="all, delete-orphan")


class FraudSignal(Base):
    __tablename__ = "fraud_signals"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("voice_sessions.id"), nullable=False)
    keyword = Column(String)
    points = Column(Integer)
    category = Column(String)
    session = relationship("VoiceSession", back_populates="fraud_signals")
```

- [ ] **Step 2: Write `db/session.py`**

```python
from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./stormvoice.db")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 3: Write `speaker_id/identify.py`**

```python
from __future__ import annotations
import numpy as np
from sqlalchemy.orm import Session
from db.models import Speaker, SpeakerEmbedding
from features.mfcc import extract_mfcc

THRESHOLD = 0.70


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def enroll_speaker(name: str, clip_paths: list[str], db: Session) -> Speaker:
    """Average MFCC vectors from all clips and store as speaker centroid."""
    vecs = [extract_mfcc(p) for p in clip_paths]
    centroid = np.mean(vecs, axis=0)

    speaker = db.query(Speaker).filter_by(name=name).first()
    if speaker is None:
        speaker = Speaker(name=name)
        db.add(speaker)
        db.flush()

    emb = db.query(SpeakerEmbedding).filter_by(speaker_id=speaker.id).first()
    if emb is None:
        emb = SpeakerEmbedding(speaker_id=speaker.id)
        db.add(emb)
    emb.set_vector(centroid.tolist())
    emb.sample_count = len(clip_paths)
    db.commit()
    db.refresh(speaker)
    return speaker


def identify_speaker(clip_path: str, db: Session) -> tuple[str, float]:
    """Cosine similarity vs stored centroids. Returns (name, confidence)."""
    vec = extract_mfcc(clip_path)
    embeddings = db.query(SpeakerEmbedding).all()
    if not embeddings:
        return "Unknown", 0.0
    best_name, best_sim = "Unknown", 0.0
    for emb in embeddings:
        centroid = np.array(emb.get_vector(), dtype=np.float32)
        sim = _cosine(vec, centroid)
        if sim > best_sim:
            best_sim = sim
            best_name = emb.speaker.name
    if best_sim < THRESHOLD:
        return "Unknown", float(best_sim)
    return best_name, float(best_sim)
```

- [ ] **Step 4: Write `tests/test_identify.py`**

```python
import numpy as np
import soundfile as sf
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base
from db.session import init_db
from speaker_id.identify import enroll_speaker, identify_speaker


@pytest.fixture
def db_session(tmp_path):
    url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()


@pytest.fixture
def sine_clip(tmp_path):
    """3-second 440 Hz sine at 16 kHz — same sound every time."""
    sr = 16000
    y = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3.0, sr * 3, endpoint=False))).astype(np.float32)
    p = tmp_path / "tone.wav"
    sf.write(p, y, sr)
    return str(p)


def test_enroll_creates_speaker(db_session, sine_clip):
    sp = enroll_speaker("baraka", [sine_clip], db_session)
    assert sp.name == "baraka"
    assert sp.embedding is not None
    assert sp.embedding.sample_count == 1


def test_enroll_upserts(db_session, sine_clip):
    enroll_speaker("baraka", [sine_clip], db_session)
    enroll_speaker("baraka", [sine_clip, sine_clip], db_session)
    count = db_session.query(__import__("db.models", fromlist=["Speaker"]).Speaker).count()
    assert count == 1  # no duplicate


def test_identify_no_speakers(db_session, sine_clip):
    name, conf = identify_speaker(sine_clip, db_session)
    assert name == "Unknown"
    assert conf == 0.0


def test_identify_returns_enrolled_speaker(db_session, sine_clip):
    enroll_speaker("baraka", [sine_clip], db_session)
    name, conf = identify_speaker(sine_clip, db_session)
    # Same clip as enrollment → very high similarity
    assert name == "baraka"
    assert conf > 0.70
```

- [ ] **Step 5: Run — expect 4 passed**

```bash
pytest tests/test_identify.py -v
```

- [ ] **Step 6: Commit**

```bash
git add db/models.py db/session.py speaker_id/identify.py tests/test_identify.py
git commit -m "feat(db,speaker_id): SQLAlchemy models + MFCC nearest-centroid enrollment/ID"
```

---

## Task 4: Fraud engine

**Files:**
- Create: `fraud/engine.py`
- Create: `tests/test_fraud.py`

**Interfaces:**
- Produces: `analyze_fraud(transcript: str) -> FraudResult`
- `FraudResult` fields: `detected_signals: list[dict]`, `risk_score: int`, `risk_level: str`, `fraud_category: str`, `recommended_action: str`.

- [ ] **Step 1: Write `tests/test_fraud.py`**

```python
import pytest
from fraud.engine import analyze_fraud


def test_clean_transcript():
    r = analyze_fraud("I want to check my transaction history.")
    assert r.risk_score == 0
    assert r.risk_level == "Low"
    assert r.detected_signals == []
    assert r.fraud_category == "Normal Request"


def test_otp_detected():
    r = analyze_fraud("Someone from customer care asked me for my OTP urgently.")
    keywords = [s["keyword"] for s in r.detected_signals]
    assert "otp" in keywords
    assert "customer care" in keywords
    assert r.risk_score >= 60   # 40 + 20
    assert r.risk_level in ("High", "Critical")


def test_pin_theft():
    r = analyze_fraud("A person called me and asked for my mobile money PIN.")
    assert r.risk_score >= 40
    assert any(s["keyword"] == "pin" for s in r.detected_signals)
    assert r.fraud_category == "PIN theft"


def test_risk_levels():
    assert analyze_fraud("").risk_level == "Low"
    assert analyze_fraud("send money to this account").risk_level == "Medium"
    assert analyze_fraud("my otp and password and send money now").risk_level in ("High", "Critical")


def test_recommended_action_high():
    r = analyze_fraud("They want my OTP and PIN right now or my account will be blocked.")
    assert r.risk_level in ("High", "Critical")
    assert "End the call" in r.recommended_action or "Block" in r.recommended_action
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_fraud.py -v
```

- [ ] **Step 3: Write `fraud/engine.py`**

```python
from __future__ import annotations
import re
from dataclasses import dataclass, field

# (phrase, points, category) — order matters: longer phrases checked first
_SIGNALS: list[tuple[str, int, str]] = [
    ("account will be blocked", 25, "Urgency pressure"),
    ("account blocked", 25, "Urgency pressure"),
    ("sim stopped working", 35, "SIM-swap risk"),
    ("sim swap", 35, "SIM-swap risk"),
    ("do not tell anyone", 30, "Secrecy pressure"),
    ("don't tell anyone", 30, "Secrecy pressure"),
    ("customer care", 20, "Fake authority"),
    ("send money", 30, "Financial manipulation"),
    ("password", 35, "Account takeover"),
    ("otp", 40, "OTP theft"),
    ("pin", 40, "PIN theft"),
]

_RISK_TABLE: list[tuple[int, str, str]] = [
    (81, "Critical", "Block immediately and contact official support."),
    (51, "High", "Do not comply. End the call and verify through official channels."),
    (21, "Medium", "Be cautious. Verify the caller's identity before sharing anything."),
    (0,  "Low",      "Allow and log. No immediate action required."),
]


@dataclass
class FraudResult:
    detected_signals: list[dict] = field(default_factory=list)
    risk_score: int = 0
    risk_level: str = "Low"
    fraud_category: str = "Normal Request"
    recommended_action: str = "Allow and log. No immediate action required."


def analyze_fraud(transcript: str) -> FraudResult:
    text = transcript.lower()
    result = FraudResult()
    seen: set[str] = set()
    category_scores: dict[str, int] = {}

    for phrase, points, category in _SIGNALS:
        if phrase in seen:
            continue
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, text):
            seen.add(phrase)
            result.detected_signals.append({"keyword": phrase, "points": points, "category": category})
            result.risk_score += points
            category_scores[category] = category_scores.get(category, 0) + points

    for threshold, level, action in _RISK_TABLE:
        if result.risk_score >= threshold:
            result.risk_level = level
            result.recommended_action = action
            break

    if category_scores:
        result.fraud_category = max(category_scores, key=lambda k: category_scores[k])

    return result
```

- [ ] **Step 4: Run — expect 5 passed**

```bash
pytest tests/test_fraud.py -v
```

- [ ] **Step 5: Commit**

```bash
git add fraud/engine.py tests/test_fraud.py
git commit -m "feat(fraud): keyword scoring engine with risk levels and categories"
```

---

## Task 5: STT (faster-whisper)

**Files:**
- Create: `stt/transcribe.py`

**Interfaces:**
- Produces: `transcribe(wav_path: str) -> str` — returns lowercase, trimmed transcript. Consumed by Task 6.
- Note: first call downloads ~145 MB model to `~/.cache/huggingface/`. This is expected and intentional.

- [ ] **Step 1: Write `stt/transcribe.py`**

```python
from __future__ import annotations
from faster_whisper import WhisperModel

_MODEL: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        # Downloads ~145 MB on first call; cached in ~/.cache/huggingface/
        _MODEL = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _MODEL


def transcribe(wav_path: str) -> str:
    """Transcribe a 16 kHz mono WAV file. Returns stripped lowercase text."""
    model = _get_model()
    segments, _ = model.transcribe(wav_path, beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()
```

- [ ] **Step 2: Smoke test (triggers model download)**

```bash
python -c "
from stt.transcribe import transcribe
import tempfile, numpy as np, soundfile as sf, os
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    sr = 16000
    y = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3.0, sr * 3))).astype(np.float32)
    sf.write(f.name, y, sr)
    result = transcribe(f.name)
    print('transcript:', repr(result))
    os.unlink(f.name)
"
```

Expected: prints something like `transcript: ''` or `transcript: 'Thank you.'` (a sine wave may produce minimal transcription — that is correct behavior for silence/tone).

- [ ] **Step 3: Commit**

```bash
git add stt/transcribe.py
git commit -m "feat(stt): faster-whisper base.en transcription wrapper"
```

---

## Task 6: FastAPI server

**Files:**
- Create: `scripts/serve.py`
- Create: `tests/test_serve.py`
- Create: `frontend/templates/index.html` (placeholder — replaced in Task 7)

**Interfaces:**
- Consumes: `identify_speaker`, `enroll_speaker` (Task 3), `analyze_fraud` (Task 4), `transcribe` (Task 5), `extract_logmel` (Task 2), DB models + `get_db` + `init_db` (Task 3).
- Produces: HTTP API at the endpoints below.

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve `frontend/templates/index.html` |
| GET | `/static/*` | Serve `frontend/static/` assets |
| POST | `/api/speakers/enroll` | `name` (form) + `clips` (files) → `{speaker_id, name, sample_count}` |
| GET | `/api/speakers` | `[{id, name, created_at}]` |
| POST | `/api/analyze` | `audio` (file) → full analysis JSON |
| GET | `/api/sessions` | `[{id, speaker, transcript, fraud_category, risk_score, risk_level, created_at}]` |
| GET | `/api/sessions/{id}` | Full session detail |

**`POST /api/analyze` response shape:**
```json
{
  "session_id": 1,
  "speaker": "baraka",
  "speaker_confidence": 0.912,
  "transcript": "someone from customer care asked me for my otp",
  "detected_signals": [{"keyword": "otp", "points": 40, "category": "OTP theft"}],
  "fraud_category": "OTP theft",
  "risk_score": 60,
  "risk_level": "High",
  "recommended_action": "Do not comply. End the call and verify through official channels.",
  "waveform_png_b64": "...",
  "spectrogram_png_b64": "..."
}
```

- [ ] **Step 1: Create placeholder `frontend/templates/index.html`**

```html
<!doctype html><html><head><title>stormVoice Lite</title></head>
<body><h1>placeholder — Task 7 replaces this</h1></body></html>
```

- [ ] **Step 2: Write `tests/test_serve.py`**

```python
import io
import json
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base
from db.session import get_db


def _wav_bytes(duration: float = 3.0) -> bytes:
    sr = 16000
    y = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("db")
    url = f"sqlite:///{tmp}/test.db"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    # Patch transcribe so tests don't download the model
    import stt.transcribe as stt_mod
    stt_mod.transcribe = lambda path: "someone asked for my otp"

    from scripts import serve
    serve.app.dependency_overrides[get_db] = override_get_db

    return TestClient(serve.app)


def test_root_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "html" in r.text.lower()


def test_list_speakers_empty(client):
    r = client.get("/api/speakers")
    assert r.status_code == 200
    assert r.json() == []


def test_enroll_speaker(client):
    wav = _wav_bytes()
    r = client.post(
        "/api/speakers/enroll",
        data={"name": "baraka"},
        files=[("clips", ("clip.wav", wav, "audio/wav"))],
    )
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "baraka"
    assert body["sample_count"] == 1


def test_list_speakers_after_enroll(client):
    r = client.get("/api/speakers")
    assert r.status_code == 200
    names = [s["name"] for s in r.json()]
    assert "baraka" in names


def test_analyze_returns_full_shape(client):
    r = client.post(
        "/api/analyze",
        files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    for key in ("session_id", "speaker", "speaker_confidence", "transcript",
                "detected_signals", "fraud_category", "risk_score", "risk_level",
                "recommended_action", "waveform_png_b64", "spectrogram_png_b64"):
        assert key in body, f"missing key: {key}"
    assert body["transcript"] == "someone asked for my otp"
    assert body["risk_score"] >= 40


def test_sessions_list(client):
    r = client.get("/api/sessions")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_session_detail(client):
    sessions = client.get("/api/sessions").json()
    sid = sessions[0]["id"]
    r = client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert "recommended_action" in r.json()


def test_session_not_found(client):
    r = client.get("/api/sessions/99999")
    assert r.status_code == 404


def test_enroll_missing_name(client):
    r = client.post(
        "/api/speakers/enroll",
        data={"name": ""},
        files=[("clips", ("clip.wav", _wav_bytes(), "audio/wav"))],
    )
    assert r.status_code == 400
```

- [ ] **Step 3: Run — expect ImportError**

```bash
pytest tests/test_serve.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.serve'`

- [ ] **Step 4: Write `scripts/serve.py`**

```python
"""stormVoice Lite — FastAPI demo server.

Loads DB at startup, serves enrollment + analysis endpoints, returns
waveform + spectrogram PNGs as base64 for the frontend.
"""
from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from db.models import FraudSignal, Speaker, VoiceSession
from db.session import get_db, init_db
from features.spectrogram import extract_logmel
from fraud.engine import analyze_fraud
from speaker_id.identify import enroll_speaker, identify_speaker
from stt.transcribe import transcribe

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND = REPO_ROOT / "frontend"

app = FastAPI(title="stormVoice Lite")


@app.on_event("startup")
def _startup() -> None:
    init_db()


if (FRONTEND / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    index = FRONTEND / "templates" / "index.html"
    if not index.exists():
        raise HTTPException(500, "index.html missing")
    return HTMLResponse(index.read_text())


# ── helpers ─────────────────────────────────────────────────────────────────

def _to_wav(raw: bytes) -> Path:
    """Decode any audio, resample to 16 kHz mono, center-crop/pad to 3 s."""
    y, _ = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
    target = 16000 * 3
    if len(y) >= target:
        start = (len(y) - target) // 2
        y = y[start:start + target]
    else:
        y = np.pad(y, (0, target - len(y)))
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(tmp, y.astype(np.float32), 16000)
    return tmp


def _png_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _waveform_png(wav: Path) -> str:
    y, sr = librosa.load(str(wav), sr=16000)
    fig, ax = plt.subplots(figsize=(6, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#4af")
    ax.set_facecolor("#111")
    fig.patch.set_facecolor("#111")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    return _png_b64(fig)


def _spectrogram_png(wav: Path) -> str:
    spec = extract_logmel(str(wav))
    fig, ax = plt.subplots(figsize=(6, 3))
    librosa.display.specshow(spec, sr=16000, x_axis="time", y_axis="mel", ax=ax)
    ax.set_facecolor("#111")
    fig.patch.set_facecolor("#111")
    return _png_b64(fig)


# ── endpoints ───────────────────────────────────────────────────────────────

@app.post("/api/speakers/enroll")
async def enroll(
    name: str = Form(...),
    clips: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> dict:
    if not name.strip():
        raise HTTPException(400, "name is required")
    if not clips:
        raise HTTPException(400, "at least one clip is required")
    tmp_paths: list[Path] = []
    try:
        for clip in clips:
            tmp_paths.append(_to_wav(await clip.read()))
        speaker = enroll_speaker(name.strip(), [str(p) for p in tmp_paths], db)
        return {"speaker_id": speaker.id, "name": speaker.name, "sample_count": len(clips)}
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


@app.get("/api/speakers")
def list_speakers(db: Session = Depends(get_db)) -> list[dict]:
    return [
        {"id": s.id, "name": s.name, "created_at": s.created_at.isoformat()}
        for s in db.query(Speaker).all()
    ]


@app.post("/api/analyze")
async def analyze(
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio")
    wav = _to_wav(raw)
    try:
        speaker_name, confidence = identify_speaker(str(wav), db)
        text = transcribe(str(wav))
        fraud = analyze_fraud(text)

        speaker_obj = db.query(Speaker).filter_by(name=speaker_name).first()
        session = VoiceSession(
            speaker_id=speaker_obj.id if speaker_obj else None,
            transcript=text,
            fraud_category=fraud.fraud_category,
            risk_score=fraud.risk_score,
            risk_level=fraud.risk_level,
            recommended_action=fraud.recommended_action,
        )
        db.add(session)
        db.flush()
        for sig in fraud.detected_signals:
            db.add(FraudSignal(session_id=session.id,
                               keyword=sig["keyword"],
                               points=sig["points"],
                               category=sig["category"]))
        db.commit()

        return {
            "session_id": session.id,
            "speaker": speaker_name,
            "speaker_confidence": round(confidence, 3),
            "transcript": text,
            "detected_signals": fraud.detected_signals,
            "fraud_category": fraud.fraud_category,
            "risk_score": fraud.risk_score,
            "risk_level": fraud.risk_level,
            "recommended_action": fraud.recommended_action,
            "waveform_png_b64": _waveform_png(wav),
            "spectrogram_png_b64": _spectrogram_png(wav),
        }
    finally:
        wav.unlink(missing_ok=True)


@app.get("/api/sessions")
def list_sessions(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.query(VoiceSession).order_by(VoiceSession.created_at.desc()).limit(50).all()
    return [
        {
            "id": s.id,
            "speaker": s.speaker.name if s.speaker else "Unknown",
            "transcript": s.transcript[:80] + ("…" if len(s.transcript) > 80 else ""),
            "fraud_category": s.fraud_category,
            "risk_score": s.risk_score,
            "risk_level": s.risk_level,
            "created_at": s.created_at.isoformat(),
        }
        for s in rows
    ]


@app.get("/api/sessions/{session_id}")
def get_session(session_id: int, db: Session = Depends(get_db)) -> dict:
    s = db.query(VoiceSession).filter_by(id=session_id).first()
    if not s:
        raise HTTPException(404, "session not found")
    return {
        "id": s.id,
        "speaker": s.speaker.name if s.speaker else "Unknown",
        "transcript": s.transcript,
        "fraud_category": s.fraud_category,
        "risk_score": s.risk_score,
        "risk_level": s.risk_level,
        "recommended_action": s.recommended_action,
        "detected_signals": [
            {"keyword": sig.keyword, "points": sig.points, "category": sig.category}
            for sig in s.fraud_signals
        ],
        "created_at": s.created_at.isoformat(),
    }
```

- [ ] **Step 5: Run — expect all tests pass**

```bash
pytest tests/test_serve.py -v
```

Expected: 9 passed.

- [ ] **Step 6: Manual smoke test**

```bash
uvicorn scripts.serve:app --reload &
sleep 3
curl -s http://127.0.0.1:8000/ | head -3
curl -s http://127.0.0.1:8000/api/speakers
kill %1
```

Expected: HTML from `/`, `[]` from `/api/speakers`.

- [ ] **Step 7: Commit**

```bash
git add scripts/serve.py tests/test_serve.py frontend/templates/index.html
git commit -m "feat(serve): FastAPI server with all 7 endpoints"
```

---

## Task 7: Frontend

**Files:**
- Create: `frontend/static/css/style.css` (copied from stormVoice — `style.css` only)
- Create: `frontend/static/fulcrum-transparent-bg.svg` (copied from stormVoice)
- Overwrite: `frontend/templates/index.html`
- Create: `frontend/static/js/app.js`

**Interfaces:**
- Consumes: `POST /api/speakers/enroll`, `GET /api/speakers`, `POST /api/analyze`, `GET /api/sessions`.
- Produces: single-page UI with Enroll tab, Analyze tab, and History tab.

- [ ] **Step 1: Copy stormVoice assets**

```bash
cp ~/stormVoice/bongo/frontend/static/css/style.css frontend/static/css/
cp ~/stormVoice/bongo/frontend/static/fulcrum-transparent-bg.svg frontend/static/
```

Verify:

```bash
ls frontend/static/css/ frontend/static/*.svg
```

- [ ] **Step 2: Write `frontend/templates/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>stormVoice Lite</title>
  <link rel="stylesheet" href="/static/css/style.css" />
  <style>
    /* Layout overrides — stormVoice base is dark/mono, we add our panels */
    body { overflow: auto; user-select: auto; font-size: 13px; }

    #app { max-width: 960px; margin: 0 auto; padding: 1.5rem 1rem 3rem; }

    header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.8rem; border-bottom: 1px solid #222; padding-bottom: 1rem; }
    header img { height: 36px; opacity: 0.85; }
    header h1 { font-size: 1rem; letter-spacing: 0.12em; color: var(--fg-dim); font-weight: 400; }

    /* Tab bar */
    .tabs { display: flex; gap: 0; margin-bottom: 1.5rem; border-bottom: 1px solid #222; }
    .tab-btn { background: none; border: none; border-bottom: 2px solid transparent;
               color: var(--fg-dim); font-family: var(--font); font-size: 0.85rem;
               letter-spacing: 0.1em; padding: 0.5rem 1.2rem; cursor: pointer; transition: var(--transition); }
    .tab-btn.active { color: var(--fg); border-bottom-color: #4af; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }

    /* Record button */
    .record-btn { display: inline-flex; align-items: center; gap: 0.5rem;
                  padding: 0.7rem 1.6rem; border-radius: 999px; border: none; cursor: pointer;
                  background: #c22; color: #fff; font-family: var(--font);
                  font-size: 0.85rem; letter-spacing: 0.08em; transition: var(--transition); }
    .record-btn.recording { background: #333; border: 1px solid #555; }
    .record-btn .dot { width: 8px; height: 8px; border-radius: 50%; background: #fff; }
    .record-btn.recording .dot { animation: blink 0.8s ease-in-out infinite; }

    /* Panels */
    .panel { background: #0a0a0a; border: 1px solid #1e1e1e; border-radius: var(--radius);
             padding: 1.2rem; margin-bottom: 1rem; }
    .panel-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
    .panel-row img { width: 100%; border-radius: 6px; background: #111; min-height: 80px; }

    /* Result fields */
    .result-grid { display: grid; grid-template-columns: max-content 1fr; gap: 0.4rem 1rem; align-items: baseline; }
    .result-label { color: var(--fg-dim); font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .result-value { color: var(--fg); }
    .result-value.speaker-name { font-size: 1.4rem; letter-spacing: 0.04em; }
    .result-value.confidence { font-size: 0.85rem; color: var(--fg-dim); }

    /* Risk badge */
    .badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px;
             font-size: 0.75rem; letter-spacing: 0.06em; font-weight: 600; }
    .badge-low { background: #1a3a1a; color: #4c4; }
    .badge-medium { background: #3a2a00; color: #fa0; }
    .badge-high { background: #3a1a00; color: #f84; }
    .badge-critical { background: #3a0000; color: #f44; }

    /* Signals */
    .signal-tag { display: inline-block; background: #1a1a2a; border: 1px solid #2a2a4a;
                  color: #88f; border-radius: 4px; padding: 0.1rem 0.4rem;
                  font-size: 0.72rem; margin: 0.15rem; }

    /* Enroll form */
    .enroll-form { display: flex; flex-direction: column; gap: 0.8rem; }
    .enroll-form input[type=text] { background: #111; border: 1px solid #2a2a2a; border-radius: 6px;
                                     color: var(--fg); font-family: var(--font); padding: 0.5rem 0.8rem;
                                     font-size: 0.9rem; width: 240px; }
    .clips-list { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.3rem; }
    .clip-item { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 4px;
                 padding: 0.2rem 0.5rem; font-size: 0.75rem; color: var(--fg-dim); }
    .submit-btn { padding: 0.5rem 1.2rem; background: #1a3a5a; border: 1px solid #2a5a8a;
                  color: #4af; border-radius: 6px; cursor: pointer; font-family: var(--font);
                  font-size: 0.82rem; letter-spacing: 0.06em; }
    .submit-btn:disabled { opacity: 0.4; cursor: not-allowed; }

    /* History */
    .history-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    .history-table th { text-align: left; color: var(--fg-dim); font-weight: 400;
                        letter-spacing: 0.08em; padding: 0.4rem 0.6rem; border-bottom: 1px solid #1e1e1e; }
    .history-table td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #0f0f0f; vertical-align: top; }
    .history-table tr:hover td { background: #0d0d0d; }
    .transcript-cell { max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--fg-dim); }

    /* Status */
    .status { font-size: 0.78rem; color: var(--fg-dim); letter-spacing: 0.06em; }
    .status.error { color: #f55; }
    .action-text { font-size: 0.85rem; color: #8f8; border-left: 2px solid #4a4; padding-left: 0.5rem; margin-top: 0.3rem; }
  </style>
</head>
<body>
<div id="app">
  <header>
    <img src="/static/fulcrum-transparent-bg.svg" alt="logo" />
    <h1>stormVoice Lite &mdash; Speaker ID &amp; Fraud Detection</h1>
  </header>

  <div class="tabs">
    <button class="tab-btn active" data-tab="analyze">Analyze</button>
    <button class="tab-btn" data-tab="enroll">Enroll Speaker</button>
    <button class="tab-btn" data-tab="history">History</button>
  </div>

  <!-- ── ANALYZE TAB ── -->
  <div id="tab-analyze" class="tab-panel active">
    <div class="panel" style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
      <button id="analyze-btn" class="record-btn"><span class="dot"></span>Record &amp; Analyze</button>
      <span id="analyze-status" class="status"></span>
    </div>

    <div class="panel-row" id="viz-row" style="display:none">
      <img id="waveform-img" alt="waveform" />
      <img id="spectrogram-img" alt="spectrogram" />
    </div>

    <div class="panel" id="result-panel" style="display:none">
      <div class="result-grid">
        <span class="result-label">Speaker</span>
        <span class="result-value speaker-name" id="r-speaker">—</span>

        <span class="result-label">Confidence</span>
        <span class="result-value confidence" id="r-confidence">—</span>

        <span class="result-label">Transcript</span>
        <span class="result-value" id="r-transcript" style="font-style:italic;color:var(--fg-dim)">—</span>

        <span class="result-label">Risk</span>
        <span class="result-value" id="r-risk">—</span>

        <span class="result-label">Category</span>
        <span class="result-value" id="r-category">—</span>

        <span class="result-label">Signals</span>
        <span class="result-value" id="r-signals">—</span>

        <span class="result-label">Action</span>
        <div class="action-text" id="r-action">—</div>
      </div>
    </div>
  </div>

  <!-- ── ENROLL TAB ── -->
  <div id="tab-enroll" class="tab-panel">
    <div class="panel">
      <div class="enroll-form">
        <div>
          <label style="color:var(--fg-dim);font-size:0.8rem;letter-spacing:0.08em">SPEAKER NAME</label><br/>
          <input type="text" id="enroll-name" placeholder="e.g. baraka" />
        </div>
        <div>
          <label style="color:var(--fg-dim);font-size:0.8rem;letter-spacing:0.08em">CLIPS (record 3)</label>
          <div class="clips-list" id="clips-list"></div>
        </div>
        <div style="display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap">
          <button id="enroll-record-btn" class="record-btn"><span class="dot"></span>Record Clip</button>
          <button id="enroll-submit-btn" class="submit-btn" disabled>Enroll Speaker</button>
          <span id="enroll-status" class="status"></span>
        </div>
      </div>
    </div>
    <div class="panel" id="speakers-panel">
      <div style="color:var(--fg-dim);font-size:0.8rem;letter-spacing:0.08em;margin-bottom:0.6rem">ENROLLED SPEAKERS</div>
      <div id="speakers-list" style="color:var(--fg-dim);font-size:0.85rem">loading…</div>
    </div>
  </div>

  <!-- ── HISTORY TAB ── -->
  <div id="tab-history" class="tab-panel">
    <div class="panel">
      <div style="color:var(--fg-dim);font-size:0.8rem;letter-spacing:0.08em;margin-bottom:0.8rem">RECENT SESSIONS</div>
      <table class="history-table">
        <thead><tr>
          <th>Speaker</th><th>Risk</th><th>Category</th><th>Transcript</th><th>Time</th>
        </tr></thead>
        <tbody id="history-body"><tr><td colspan="5" style="color:var(--fg-dim)">loading…</td></tr></tbody>
      </table>
    </div>
  </div>
</div>

<script src="/static/js/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write `frontend/static/js/app.js`**

```javascript
/* stormVoice Lite — vanilla JS frontend */

const RECORD_MS = 4000;

// ── Tab routing ──────────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "history") loadHistory();
    if (btn.dataset.tab === "enroll") loadSpeakers();
  });
});

// ── MediaRecorder helper ─────────────────────────────────────────────────────
async function recordClip(durationMs) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  return new Promise((resolve, reject) => {
    const chunks = [];
    const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mr.ondataavailable = (e) => chunks.push(e.data);
    mr.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      resolve(new Blob(chunks, { type: "audio/webm" }));
    };
    mr.onerror = reject;
    mr.start();
    setTimeout(() => mr.stop(), durationMs);
  });
}

// ── Analyze tab ──────────────────────────────────────────────────────────────
const analyzeBtn = document.getElementById("analyze-btn");
const analyzeStatus = document.getElementById("analyze-status");

analyzeBtn.addEventListener("click", async () => {
  if (analyzeBtn.classList.contains("recording")) return;
  analyzeBtn.classList.add("recording");
  analyzeBtn.querySelector("span.dot").style.background = "#f55";
  analyzeStatus.textContent = `Recording ${RECORD_MS / 1000}s…`;
  analyzeStatus.className = "status";

  try {
    const blob = await recordClip(RECORD_MS);
    analyzeStatus.textContent = "Analyzing…";
    const fd = new FormData();
    fd.append("audio", blob, "clip.webm");
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderResult(data);
    analyzeStatus.textContent = "";
  } catch (err) {
    analyzeStatus.textContent = "Error: " + err.message;
    analyzeStatus.className = "status error";
  } finally {
    analyzeBtn.classList.remove("recording");
  }
});

function badgeClass(level) {
  return { Low: "badge-low", Medium: "badge-medium", High: "badge-high", Critical: "badge-critical" }[level] || "badge-low";
}

function renderResult(data) {
  document.getElementById("waveform-img").src = "data:image/png;base64," + data.waveform_png_b64;
  document.getElementById("spectrogram-img").src = "data:image/png;base64," + data.spectrogram_png_b64;
  document.getElementById("viz-row").style.display = "grid";

  document.getElementById("r-speaker").textContent = data.speaker;
  document.getElementById("r-confidence").textContent = `${(data.speaker_confidence * 100).toFixed(1)}%`;
  document.getElementById("r-transcript").textContent = data.transcript || "(no speech detected)";
  document.getElementById("r-risk").innerHTML =
    `<span class="badge ${badgeClass(data.risk_level)}">${data.risk_level}</span> &nbsp; score: ${data.risk_score}`;
  document.getElementById("r-category").textContent = data.fraud_category;
  document.getElementById("r-signals").innerHTML = data.detected_signals.length
    ? data.detected_signals.map((s) => `<span class="signal-tag">${s.keyword} +${s.points}</span>`).join("")
    : "<span style='color:var(--fg-dim)'>none</span>";
  document.getElementById("r-action").textContent = data.recommended_action;
  document.getElementById("result-panel").style.display = "block";
}

// ── Enroll tab ───────────────────────────────────────────────────────────────
const enrollRecordBtn = document.getElementById("enroll-record-btn");
const enrollSubmitBtn = document.getElementById("enroll-submit-btn");
const enrollStatus = document.getElementById("enroll-status");
const clipsList = document.getElementById("clips-list");
const enrolledBlobs = [];

enrollRecordBtn.addEventListener("click", async () => {
  if (enrollRecordBtn.classList.contains("recording")) return;
  if (enrolledBlobs.length >= 3) {
    enrollStatus.textContent = "3 clips already recorded.";
    return;
  }
  enrollRecordBtn.classList.add("recording");
  enrollStatus.textContent = `Recording clip ${enrolledBlobs.length + 1}/3…`;
  try {
    const blob = await recordClip(RECORD_MS);
    enrolledBlobs.push(blob);
    const tag = document.createElement("span");
    tag.className = "clip-item";
    tag.textContent = `clip ${enrolledBlobs.length}`;
    clipsList.appendChild(tag);
    enrollStatus.textContent = enrolledBlobs.length < 3 ? `${enrolledBlobs.length}/3 recorded` : "Ready to enroll";
    enrollSubmitBtn.disabled = enrolledBlobs.length === 0;
  } catch (err) {
    enrollStatus.textContent = "Mic error: " + err.message;
    enrollStatus.className = "status error";
  } finally {
    enrollRecordBtn.classList.remove("recording");
  }
});

enrollSubmitBtn.addEventListener("click", async () => {
  const name = document.getElementById("enroll-name").value.trim();
  if (!name) { enrollStatus.textContent = "Enter a name first."; return; }
  if (enrolledBlobs.length === 0) { enrollStatus.textContent = "Record at least one clip."; return; }
  enrollStatus.textContent = "Enrolling…";
  enrollSubmitBtn.disabled = true;
  const fd = new FormData();
  fd.append("name", name);
  enrolledBlobs.forEach((b, i) => fd.append("clips", b, `clip${i}.webm`));
  try {
    const res = await fetch("/api/speakers/enroll", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    enrollStatus.textContent = `✓ Enrolled "${data.name}" (${data.sample_count} clip${data.sample_count > 1 ? "s" : ""})`;
    enrollStatus.className = "status";
    enrolledBlobs.length = 0;
    clipsList.innerHTML = "";
    document.getElementById("enroll-name").value = "";
    await loadSpeakers();
  } catch (err) {
    enrollStatus.textContent = "Error: " + err.message;
    enrollStatus.className = "status error";
    enrollSubmitBtn.disabled = false;
  }
});

async function loadSpeakers() {
  const el = document.getElementById("speakers-list");
  try {
    const res = await fetch("/api/speakers");
    const data = await res.json();
    el.innerHTML = data.length
      ? data.map((s) => `<span class="clip-item" style="margin:0.15rem">${s.name}</span>`).join("")
      : "<span style='color:var(--fg-dim)'>none enrolled yet</span>";
  } catch { el.textContent = "error loading speakers"; }
}

// ── History tab ──────────────────────────────────────────────────────────────
async function loadHistory() {
  const body = document.getElementById("history-body");
  try {
    const res = await fetch("/api/sessions");
    const rows = await res.json();
    if (!rows.length) {
      body.innerHTML = "<tr><td colspan='5' style='color:var(--fg-dim)'>no sessions yet</td></tr>";
      return;
    }
    body.innerHTML = rows.map((s) => `
      <tr>
        <td>${s.speaker}</td>
        <td><span class="badge ${badgeClass(s.risk_level)}">${s.risk_level}</span></td>
        <td style="font-size:0.78rem;color:var(--fg-dim)">${s.fraud_category}</td>
        <td class="transcript-cell">${s.transcript || "—"}</td>
        <td style="font-size:0.72rem;color:var(--fg-dim)">${new Date(s.created_at).toLocaleTimeString()}</td>
      </tr>`).join("");
  } catch { body.innerHTML = "<tr><td colspan='5' style='color:var(--fg-dim)'>error loading history</td></tr>"; }
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): stormVoice fork + analyze/enroll/history single-page UI"
```

---

## Task 8: README + final verification

**Files:**
- Overwrite: `README.md`

- [ ] **Step 1: Run full test suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Overwrite `README.md`**

```markdown
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

1. **Enroll tab** — record 3 clips per group member. Each enrolled speaker becomes a known identity.
2. **Analyze tab** — record a clip. Receive: speaker name, confidence, transcript, fraud signals, risk level, recommended action.
3. **History tab** — browse all past analysis sessions.

## Swapping in real teammate data

Enrollment is live — teammates just open the browser, go to **Enroll**, and record 3 clips. No code changes, no retraining.

## Stack

| Layer | Choice |
|-------|--------|
| Backend | FastAPI + uvicorn |
| Speaker ID | MFCC (120,) nearest-centroid |
| STT | faster-whisper base.en |
| Fraud engine | Rule-based keyword scoring |
| Database | SQLite + SQLAlchemy |
| Frontend | Vanilla JS + stormVoice CSS |

## Tests

```bash
pytest -v
```

## Swap to PostgreSQL

Set `DATABASE_URL=postgresql://user:pass@host/dbname` before starting the server. No other changes needed.
```

- [ ] **Step 3: Full end-to-end smoke test**

```bash
uvicorn scripts.serve:app --reload &
sleep 4
curl -s http://127.0.0.1:8000/api/speakers
curl -s http://127.0.0.1:8000/api/sessions
kill %1
```

Expected: both return `[]` (empty DB on fresh start), server starts without errors.

- [ ] **Step 4: Final commit**

```bash
git add README.md
git commit -m "docs: quickstart README for stormVoice Lite"
```

---

## Deliverables after Task 8

- `pytest -v` — all tests green.
- `uvicorn scripts.serve:app --reload` starts cleanly.
- Browser at `http://127.0.0.1:8000` shows the dark-themed dashboard.
- Enroll a speaker → analyze a clip → see result with waveform, spectrogram, risk badge.
- Session history persisted in `stormvoice.db`.
- Teammates clone repo, `pip install -r requirements.txt`, run uvicorn, enroll themselves — no code changes needed.
