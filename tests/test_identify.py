import numpy as np
import soundfile as sf
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, Speaker
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
    count = db_session.query(Speaker).count()
    assert count == 1


def test_identify_no_speakers(db_session, sine_clip):
    name, conf = identify_speaker(sine_clip, db_session)
    assert name == "Unknown"
    assert conf == 0.0


def test_identify_returns_enrolled_speaker(db_session, sine_clip):
    enroll_speaker("baraka", [sine_clip], db_session)
    name, conf = identify_speaker(sine_clip, db_session)
    assert name == "baraka"
    assert conf > 0.70
