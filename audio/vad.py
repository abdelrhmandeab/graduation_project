import torch
from silero_vad import get_speech_timestamps, read_audio

model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)

(get_speech_timestamps, _, read_audio, _, _) = utils

def is_speech(audio_path):
    wav = read_audio(audio_path, sampling_rate=16000)
    timestamps = get_speech_timestamps(wav, model)
    return len(timestamps) > 0
