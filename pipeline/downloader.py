import io
import os
import queue
import threading

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

DATASET_NAME = "yanirmr/VoxKnesset"
TARGET_SR = 16000


class Downloader(threading.Thread):
    def __init__(self, q: queue.Queue, skip_ids: set, max_items: int | None = None):
        super().__init__(daemon=True)
        self.q = q
        self.skip_ids = skip_ids
        self.max_items = max_items

    def run(self):
        try:
            ds = load_dataset(DATASET_NAME, split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))

            delivered = 0
            for sample in ds:
                if self.max_items is not None and delivered >= self.max_items:
                    break

                filename = os.path.basename(sample["audio"]["path"])
                if filename in self.skip_ids:
                    continue

                try:
                    audio_bytes = sample["audio"]["bytes"]
                    array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
                    array = array.astype(np.float32)

                    if sampling_rate != TARGET_SR:
                        import librosa
                        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=TARGET_SR)
                except Exception as e:
                    print(f"[downloader] skipping {filename}: {e}")
                    continue

                self.q.put((filename, array))
                delivered += 1
        finally:
            self.q.put(None)
