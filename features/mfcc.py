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
