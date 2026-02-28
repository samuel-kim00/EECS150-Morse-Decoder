import argparse
import numpy as np

from src.fft_analysis import compute_fft
from src.tone_activity import band_energy_fft
from src.morse_map import decode_symbol
from src.wav_input import load_wav_mono_resampled

FS_DEFAULT = 44100
FC = 800.0
BW = 180.0
FRAME_SEC = 0.02

WAV_SKIP_SEC = 0.00
SCAN_SEC = 4.0


def compute_energy_series(x, fs):
    frame_len = int(fs * FRAME_SEC)
    n_frames = len(x) // frame_len
    energies = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        fr = x[i * frame_len : (i + 1) * frame_len]
        freqs, mag = compute_fft(fr, fs)
        energies[i] = band_energy_fft(freqs, mag, fc=FC, bw=BW)

    return energies


def pick_threshold(energies):
    e = np.asarray(energies, dtype=np.float32)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return 0.0

    lo = float(np.percentile(e, 10))
    hi = float(np.percentile(e, 90))
    thr = 0.5 * (lo + hi)
    return thr


def run_length_encode(bits):
    if len(bits) == 0:
        return []
    runs = []
    cur = bits[0]
    count = 1
    for b in bits[1:]:
        if b == cur:
            count += 1
        else:
            runs.append((cur, count))
            cur = b
            count = 1
    runs.append((cur, count))
    return runs


def decode_offline_wav(x, fs):
    energies = compute_energy_series(x, fs)
    thr = pick_threshold(energies)

    on = energies >= thr

    runs = run_length_encode(on.tolist())

    on_lens = np.array([cnt for val, cnt in runs if val], dtype=np.float32)
    if on_lens.size == 0:
        return ""

    dot_frames = float(np.percentile(on_lens, 30))
    dot_frames = max(dot_frames, 1.0)
    dash_frames = 3.0 * dot_frames

    intra_off = 1.5 * dot_frames
    letter_off = 3.0 * dot_frames
    word_off = 7.0 * dot_frames

    current = ""
    out = []

    def flush_letter():
        nonlocal current
        if current != "":
            out.append(decode_symbol(current))
            current = ""

    for val, cnt in runs:
        if val:
            if cnt < 2.0 * dot_frames:
                current += "."
            else:
                current += "-"
        else:
            if cnt < intra_off:
                continue
            if cnt >= word_off:
                flush_letter()
                if len(out) == 0 or out[-1] != " ":
                    out.append(" ")
            elif cnt >= letter_off:
                flush_letter()

    flush_letter()
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True)
    args = ap.parse_args()

    fs, x = load_wav_mono_resampled(args.wav, FS_DEFAULT)

    skip = int(WAV_SKIP_SEC * fs)
    if 0 < skip < len(x):
        x = x[skip:]

    print(f"WAV loaded: {args.wav} (fs={fs}, N={len(x)})")

    text = decode_offline_wav(x, fs)
    print("Final Decoded Text:", text)


if __name__ == "__main__":
    main()
