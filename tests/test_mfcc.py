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
