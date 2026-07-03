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
