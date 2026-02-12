import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time

SAMPLE_RATE = 16000

def record_until_silence(
    filename="input.wav",
    max_duration=10
):
    print("🎤 Listening...")
    audio_buffer = []
    start_time = time.time()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    ) as stream:
        while True:
            data, _ = stream.read(1024)
            audio_buffer.append(data)

            if time.time() - start_time > max_duration:
                break

    audio = np.concatenate(audio_buffer)
    wav.write(filename, SAMPLE_RATE, audio)
    print("✅ Audio captured")
