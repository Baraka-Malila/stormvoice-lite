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
