import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms

# Load pre-trained wake word model
model = Model(wakeword_models=["hey_jarvis"])

def listen_for_wake_word():
    print("🟢 Waiting for wake word... (say 'Jarvis')")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    ) as stream:
        while True:
            audio_chunk, _ = stream.read(CHUNK_SIZE)
            audio_chunk = audio_chunk.flatten().astype(np.float32) / 32768.0

            prediction = model.predict(audio_chunk)

            if prediction["hey_jarvis"] > 0.6:
                print("🟡 Wake word detected!")
                return
