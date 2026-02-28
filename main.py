import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

from src.fft_analysis import compute_fft
from src.tone_activity import band_energy_fft, calibrate_threshold

FS = 44100
FC = 800.0
BW = 60.0
FRAME_SEC = 0.05
CALIB_SEC = 2.0

def now():
    return time.time()

def rec_frame(frame_len):
    x = sd.rec(frame_len, samplerate=FS, channels=1, dtype="float32")
    try:
        sd.wait()
    except KeyboardInterrupt:
        try:
            sd.stop()
        except Exception:
            pass
        raise
    return x[:, 0]

def main():
    frame_len = int(FS * FRAME_SEC)

    print("EECS150 Morse Tone Activity Detector")
    print(f"FS={FS} Hz, FC={FC} Hz, BW={BW} Hz, frame={FRAME_SEC*1000:.0f} ms")
    print("Calibration: stay quiet... (Ctrl+C to stop)")

    cal_frames = int(CALIB_SEC / FRAME_SEC)
    cal_energies = []

    try:
        for _ in range(cal_frames):
            x = rec_frame(frame_len)
            freqs, mag = compute_fft(x, FS)
            cal_energies.append(band_energy_fft(freqs, mag, fc=FC, bw=BW))
    except KeyboardInterrupt:
        print("Stopped during calibration.")
        return

    thr = calibrate_threshold(cal_energies, k=4.0)
    print(f"Threshold = {thr:.2f}")
    print("Play 800Hz tone. Ctrl+C to stop.")

    state = "OFF"
    state_start = now()

    energy_hist = []
    t_hist = []

    try:
        while True:
            x = rec_frame(frame_len)
            freqs, mag = compute_fft(x, FS)
            e = band_energy_fft(freqs, mag, fc=FC, bw=BW)

            t_hist.append(now())
            energy_hist.append(e)

            new_state = "ON" if e >= thr else "OFF"

            if new_state != state:
                dur = now() - state_start
                print(f"{state} duration: {dur:.3f} sec -> {new_state}")
                state = new_state
                state_start = now()

            max_len = int(10 / FRAME_SEC)
            if len(energy_hist) > max_len:
                energy_hist = energy_hist[-max_len:]
                t_hist = t_hist[-max_len:]

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if len(energy_hist) >= 5:
            t0 = t_hist[0]
            tt = [ti - t0 for ti in t_hist]
            plt.plot(tt, energy_hist)
            plt.axhline(thr, linestyle="--")
            plt.xlabel("Time (sec)")
            plt.ylabel("Band Energy")
            plt.title("800Hz Band Energy vs Time")
            plt.tight_layout()
            plt.savefig("energy_timeline.png")
            plt.close()
            print("Saved energy_timeline.png")

if __name__ == "__main__":
    main()
