import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

def _to_float32_mono(raw):
    if raw.ndim == 2:
        raw = raw[:, 0]

    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        x = raw.astype(np.float32) / float(max(abs(info.min), info.max))
    else:
        x = raw.astype(np.float32)
        m = float(np.max(np.abs(x)))
        if m > 0:
            x = x / m

    x = x - float(np.mean(x))
    return x

def load_wav_mono(path: str):
    fs, raw = wavfile.read(path)
    x = _to_float32_mono(raw)
    return fs, x

def load_wav_mono_resampled(path: str, target_fs: int):
    fs, x = load_wav_mono(path)
    if fs == target_fs:
        return fs, x

    g = gcd(fs, target_fs)
    up = target_fs // g
    down = fs // g
    y = resample_poly(x, up, down).astype(np.float32)
    y = y - float(np.mean(y))
    return target_fs, y
