import numpy as np
from scipy.fft import fft

def compute_fft(x, fs):
    N = len(x)
    X = fft(x)
    freqs = np.fft.fftfreq(N, 1/fs)
    mag = np.abs(X)
    return freqs, mag
