"""Integration tests for the FastAPI server.

Uses an in-memory SQLite DB and skips model loading so no trained models
or whisper downloads are needed during CI.
"""
import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base
from db.session import get_db


def _wav_bytes(duration: float = 3.0) -> bytes:
    sr = 16_000
    t  = np.linspace(0, duration, int(sr * duration))
    y  = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    engine  = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    # Patch STT so the test never downloads the whisper model
    import stt.transcribe as stt_mod
    stt_mod.transcribe = lambda path: "please give me your otp"

    from scripts import serve

    # Inject stub SVM and CNN so /api/analyze doesn't 503
    import joblib
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    import torch
    from models.cnn import SpeakerCNN

    le   = LabelEncoder().fit(["baraka", "speaker2"])
    feat = np.array([[1.0] * 120, [0.5] * 120])
    lab  = np.array([0, 1])
    pipe = Pipeline([("sc", StandardScaler()), ("svm", SVC(probability=True))])
    pipe.fit(feat, lab)

    serve._svm_model   = pipe
    serve._svm_classes = list(le.classes_)
    serve._cnn_classes = list(le.classes_)
    cnn = SpeakerCNN(n_classes=2)
    cnn.eval()
    serve._cnn_model = cnn

    serve.app.dependency_overrides[get_db] = override_get_db
    return TestClient(serve.app)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "stormvoice" in r.text.lower()


def test_status(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert "svm_loaded" in body
    assert "cnn_loaded" in body


def test_sessions_empty(client):
    r = client.get("/api/sessions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_analyze(client):
    r = client.post(
        "/api/analyze",
        files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    for key in ("session_id", "svm_speaker", "cnn_speaker", "transcript",
                "detected_signals", "fraud_category", "risk_score", "risk_level",
                "recommended_action", "waveform_png_b64", "spectrogram_png_b64"):
        assert key in body, f"missing: {key}"
    assert body["transcript"] == "please give me your otp"
    assert body["risk_score"] >= 40


def test_sessions_after_analyze(client):
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


def test_analyze_empty(client):
    r = client.post(
        "/api/analyze",
        files={"audio": ("clip.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400
