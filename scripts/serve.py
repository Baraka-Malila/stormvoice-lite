"""stormVoice Lite — FastAPI server.

Three-part demo:
  1. Enroll  — POST /api/speakers/enroll  saves clips + retrains SVM live
  2. Analyze — POST /api/analyze          SVM + CNN speaker-ID, STT, fraud
  3. History — GET  /api/sessions         past sessions

Models loaded at startup from models/svm.joblib and models/cnn.pt.
Run notebooks to (re-)train. Enrollment retrains SVM automatically.
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import librosa
import librosa.display
import matplotlib
import numpy as np
import soundfile as sf
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from db.models import FraudSignal, VoiceSession
from db.session import get_db, init_db
from features.mfcc import extract_mfcc
from features.spectrogram import extract_logmel
from fraud.engine import analyze_fraud
from models.cnn import SpeakerCNN
from stt.transcribe import transcribe

REPO_ROOT    = Path(__file__).resolve().parent.parent
FRONTEND     = REPO_ROOT / "frontend"
MODELS_DIR   = REPO_ROOT / "models"
RECORD_DIR   = REPO_ROOT / "data" / "recordings"
RECORD_DIR.mkdir(parents=True, exist_ok=True)

# ── Model state (protected by _model_lock for live retraining) ────────────────
_model_lock  = threading.Lock()
_svm_model:   Optional[Pipeline]    = None
_svm_classes: list[str]             = []
_cnn_model:   Optional[SpeakerCNN]  = None
_cnn_classes: list[str]             = []


def _load_models() -> None:
    global _svm_model, _svm_classes, _cnn_model, _cnn_classes
    svm_p   = MODELS_DIR / "svm.joblib"
    svm_cls = MODELS_DIR / "svm_classes.json"
    if svm_p.exists() and svm_cls.exists():
        _svm_model   = joblib.load(svm_p)
        _svm_classes = json.loads(svm_cls.read_text())
        print(f"[serve] SVM loaded — {_svm_classes}")
    else:
        print("[serve] SVM not found — use Enroll tab or run notebook 01")

    cnn_p   = MODELS_DIR / "cnn.pt"
    cnn_cls = MODELS_DIR / "cnn_classes.json"
    if cnn_p.exists() and cnn_cls.exists():
        _cnn_classes = json.loads(cnn_cls.read_text())
        _cnn_model   = SpeakerCNN(n_classes=len(_cnn_classes))
        _cnn_model.load_state_dict(torch.load(cnn_p, map_location="cpu"))
        _cnn_model.eval()
        print(f"[serve] CNN loaded — {_cnn_classes}")
    else:
        print("[serve] CNN not found — run notebook 02")


def _retrain_svm() -> tuple[list[str], int]:
    """Retrain SVM on all data in RECORD_DIR. Updates global model in-place."""
    global _svm_model, _svm_classes
    X, y = [], []
    for spk_dir in sorted(RECORD_DIR.iterdir()):
        if not spk_dir.is_dir():
            continue
        wavs = sorted(spk_dir.glob("*.wav"))
        for wav in wavs:
            try:
                X.append(extract_mfcc(str(wav)))
                y.append(spk_dir.name)
            except Exception:
                pass

    if len(set(y)) < 2:
        raise ValueError(f"Need at least 2 speakers to train SVM (found: {set(y) or 'none'})")

    X_arr = np.array(X, dtype=np.float32)
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, C=10, gamma="scale")),
    ])
    pipe.fit(X_arr, y_enc)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "svm.joblib")
    (MODELS_DIR / "svm_classes.json").write_text(json.dumps(list(le.classes_)))

    with _model_lock:
        _svm_model   = pipe
        _svm_classes = list(le.classes_)

    return list(le.classes_), len(X)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _load_models()
    yield


app = FastAPI(title="stormVoice Lite", lifespan=lifespan)

if (FRONTEND / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    idx = FRONTEND / "templates" / "index.html"
    if not idx.exists():
        raise HTTPException(500, "index.html missing")
    return HTMLResponse(idx.read_text())


@app.get("/api/status")
def status() -> dict:
    return {
        "svm_loaded":  _svm_model is not None,
        "cnn_loaded":  _cnn_model is not None,
        "svm_classes": _svm_classes,
        "cnn_classes": _cnn_classes,
    }


@app.get("/api/speakers")
def list_speakers() -> list[str]:
    return sorted(
        d.name for d in RECORD_DIR.iterdir()
        if d.is_dir() and list(d.glob("*.wav"))
    )


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _to_wav(raw: bytes) -> Path:
    """Decode any audio, resample to 16 kHz mono, crop/pad to 3 s."""
    suffix = ".wav" if raw[:4] == b"RIFF" else ".webm"
    tmp_in = Path(tempfile.mkstemp(suffix=suffix)[1])
    tmp_in.write_bytes(raw)
    try:
        y, _ = librosa.load(str(tmp_in), sr=16_000, mono=True)
    finally:
        tmp_in.unlink(missing_ok=True)

    target = 16_000 * 3
    if len(y) >= target:
        start = (len(y) - target) // 2
        y = y[start : start + target]
    else:
        y = np.pad(y, (0, target - len(y)))

    out = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(out, y.astype(np.float32), 16_000)
    return out


def _png_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _waveform_png(wav: Path) -> str:
    y, sr = librosa.load(str(wav), sr=16_000)
    fig, ax = plt.subplots(figsize=(6, 1.8))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#fff", alpha=0.7)
    ax.set_facecolor("#000"); fig.patch.set_facecolor("#000")
    ax.tick_params(colors="#555"); ax.set_xlabel(""); ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_edgecolor("#222")
    return _png_b64(fig)


def _spectrogram_png(wav: Path) -> str:
    spec = extract_logmel(str(wav))
    fig, ax = plt.subplots(figsize=(6, 2.4))
    librosa.display.specshow(spec, sr=16_000, x_axis="time", y_axis="mel",
                             ax=ax, cmap="magma")
    ax.set_facecolor("#000"); fig.patch.set_facecolor("#000")
    ax.set_xlabel(""); ax.set_ylabel("")
    return _png_b64(fig)


# ── Classification ────────────────────────────────────────────────────────────

def _svm_predict(wav: Path) -> tuple[Optional[str], Optional[float]]:
    with _model_lock:
        model, classes = _svm_model, _svm_classes
    if model is None:
        return None, None
    feat    = extract_mfcc(str(wav)).reshape(1, -1)
    speaker = model.predict(feat)[0]
    conf    = float(model.predict_proba(feat)[0].max())
    # speaker is the encoded label; decode via classes
    if isinstance(speaker, (int, np.integer)):
        speaker = classes[speaker]
    return str(speaker), conf


def _cnn_predict(wav: Path) -> tuple[Optional[str], Optional[float]]:
    if _cnn_model is None:
        return None, None
    mel  = extract_logmel(str(wav))
    x    = torch.tensor(mel).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = _cnn_model(x)
    probs = torch.softmax(logits, dim=1)[0]
    idx   = int(probs.argmax())
    return _cnn_classes[idx], float(probs[idx])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/speakers/enroll")
async def enroll(
    name: str = Form(...),
    clips: list[UploadFile] = File(...),
) -> dict:
    name = name.strip()
    if not name:
        raise HTTPException(400, "name is required")
    if not clips:
        raise HTTPException(400, "at least one clip is required")

    spk_dir = RECORD_DIR / name
    spk_dir.mkdir(parents=True, exist_ok=True)

    # Count existing clips to avoid overwriting
    existing = len(list(spk_dir.glob("*.wav")))
    saved = 0
    tmp_paths: list[Path] = []
    try:
        for i, clip in enumerate(clips):
            raw = await clip.read()
            if not raw:
                continue
            wav = _to_wav(raw)
            tmp_paths.append(wav)
            dest = spk_dir / f"{existing + i:04d}.wav"
            wav.rename(dest)
            saved += 1
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)

    if saved == 0:
        raise HTTPException(400, "no valid clips received")

    # Retrain SVM with all recordings (fast: <10 s for small datasets)
    try:
        classes, total_clips = _retrain_svm()
        return {
            "speaker":     name,
            "clips_saved": saved,
            "svm_classes": classes,
            "total_clips": total_clips,
            "message":     f"SVM retrained on {total_clips} clips across {len(classes)} speakers.",
        }
    except ValueError as e:
        return {
            "speaker":     name,
            "clips_saved": saved,
            "svm_classes": [],
            "total_clips": saved,
            "message":     str(e) + " (need ≥2 enrolled speakers before SVM can train)",
        }


@app.post("/api/analyze")
async def analyze(
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio")

    with _model_lock:
        no_models = _svm_model is None and _cnn_model is None
    if no_models:
        raise HTTPException(
            503,
            "No speaker models loaded. Enroll at least 2 speakers first "
            "(Enroll tab), then the SVM trains automatically.",
        )

    wav = _to_wav(raw)
    try:
        svm_speaker, svm_conf = _svm_predict(wav)
        cnn_speaker, cnn_conf = _cnn_predict(wav)
        text  = transcribe(str(wav))
        fraud = analyze_fraud(text)

        session = VoiceSession(
            svm_speaker=svm_speaker, svm_confidence=svm_conf,
            cnn_speaker=cnn_speaker, cnn_confidence=cnn_conf,
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
            "session_id":      session.id,
            "svm_speaker":     svm_speaker,
            "svm_confidence":  round(svm_conf, 3) if svm_conf else None,
            "cnn_speaker":     cnn_speaker,
            "cnn_confidence":  round(cnn_conf, 3) if cnn_conf else None,
            "transcript":      text,
            "detected_signals": fraud.detected_signals,
            "fraud_category":  fraud.fraud_category,
            "risk_score":      fraud.risk_score,
            "risk_level":      fraud.risk_level,
            "recommended_action": fraud.recommended_action,
            "waveform_png_b64":    _waveform_png(wav),
            "spectrogram_png_b64": _spectrogram_png(wav),
        }
    finally:
        wav.unlink(missing_ok=True)


@app.get("/api/sessions")
def list_sessions(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.query(VoiceSession).order_by(VoiceSession.created_at.desc()).limit(50).all()
    return [
        {
            "id":           s.id,
            "svm_speaker":  s.svm_speaker or "—",
            "cnn_speaker":  s.cnn_speaker or "—",
            "transcript":   (s.transcript or "")[:80] + ("…" if len(s.transcript or "") > 80 else ""),
            "fraud_category": s.fraud_category,
            "risk_level":   s.risk_level,
            "created_at":   s.created_at.isoformat(),
        }
        for s in rows
    ]


@app.get("/api/sessions/{session_id}")
def get_session(session_id: int, db: Session = Depends(get_db)) -> dict:
    s = db.query(VoiceSession).filter_by(id=session_id).first()
    if not s:
        raise HTTPException(404, "session not found")
    return {
        "id":              s.id,
        "svm_speaker":     s.svm_speaker,
        "svm_confidence":  s.svm_confidence,
        "cnn_speaker":     s.cnn_speaker,
        "cnn_confidence":  s.cnn_confidence,
        "transcript":      s.transcript,
        "fraud_category":  s.fraud_category,
        "risk_score":      s.risk_score,
        "risk_level":      s.risk_level,
        "recommended_action": s.recommended_action,
        "detected_signals": [
            {"keyword": sig.keyword, "points": sig.points, "category": sig.category}
            for sig in s.fraud_signals
        ],
        "created_at": s.created_at.isoformat(),
    }
