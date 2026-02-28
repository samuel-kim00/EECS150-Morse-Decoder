import numpy as np

TARGET = 800
TOL = 20

def detect_tone(freqs, mag):
    idx = np.argmax(mag)
    peak = abs(freqs[idx])
    return (abs(peak - TARGET) <= TOL), peak
