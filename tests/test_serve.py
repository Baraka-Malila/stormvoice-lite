import io
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

    # Patch transcribe so tests don't trigger model download
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
    # FastAPI may return 422 (form validation) or 400 (our guard) — both are correct rejections
    assert r.status_code in (400, 422)
