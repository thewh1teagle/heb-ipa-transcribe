import queue
import threading

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

MAX_CHUNK_S = 25
SR = 16000


class Preprocessor(threading.Thread):
    def __init__(self, in_q: queue.Queue, out_q: queue.Queue):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.out_q = out_q

    def run(self):
        try:
            vad_model = load_silero_vad()
            while True:
                item = self.in_q.get()
                if item is None:
                    break

                filename, audio = item
                try:
                    chunks = _get_chunks(audio, vad_model)
                except Exception as e:
                    print(f"[preprocessor] skipping {filename}: {e}")
                    continue

                if chunks:
                    self.out_q.put((filename, chunks))
        finally:
            self.out_q.put(None)


def _get_chunks(audio: np.ndarray, vad_model) -> list[np.ndarray]:
    wav_tensor = torch.from_numpy(audio)
    timestamps = get_speech_timestamps(wav_tensor, vad_model, return_seconds=True, sampling_rate=SR)

    if not timestamps:
        return []

    chunks = []
    for ts in timestamps:
        chunk_start = int(ts["start"] * SR)
        chunk_end = int(ts["end"] * SR)
        max_samples = MAX_CHUNK_S * SR
        while chunk_end - chunk_start > max_samples:
            chunks.append((chunk_start, chunk_start + max_samples))
            chunk_start += max_samples
        if chunk_end > chunk_start:
            chunks.append((chunk_start, chunk_end))

    merged = []
    current_start, current_end = chunks[0]
    for chunk_start, chunk_end in chunks[1:]:
        if (chunk_end - current_start) <= MAX_CHUNK_S * SR:
            current_end = chunk_end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = chunk_start, chunk_end
    merged.append((current_start, current_end))

    return [(s / SR, e / SR, audio[s:e]) for s, e in merged]
