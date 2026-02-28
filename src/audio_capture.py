import sounddevice as sd

FS = 44100

def record_audio(duration=1.0):
    audio = sd.rec(int(duration * FS), samplerate=FS, channels=1)
    sd.wait()
    return audio[:, 0]
