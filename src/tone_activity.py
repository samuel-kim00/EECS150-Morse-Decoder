import numpy as np

def band_energy_fft(freqs, mag, fc=800.0, bw=40.0):
    f = np.abs(freqs)
    lo = fc - bw / 2
    hi = fc + bw / 2
    mask = (f >= lo) & (f <= hi)
    return float(np.sum(mag[mask]))

def calibrate_threshold(energies, k=3.0):
    e = np.array(energies, dtype=float)
    med = np.median(e)
    mad = np.median(np.abs(e - med)) + 1e-9
    return float(med + k * mad)
