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
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="stormVoice Lite", lifespan=lifespan)


if (FRONTEND / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    index = FRONTEND / "templates" / "index.html"
    if not index.exists():
        raise HTTPException(500, "index.html missing")
    return HTMLResponse(index.read_text())


# ── helpers ──────────────────────────────────────────────────────────────────

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


# ── endpoints ─────────────────────────────────────────────────────────────────

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
