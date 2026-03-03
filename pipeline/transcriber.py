import queue

import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad

IPA_MODEL_ID = "thewh1teagle/whisper-heb-ipa-large-v3-turbo-ct2"
TEXT_MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEVICE = "cuda"
COMPUTE_TYPE = "int8"
LANGUAGE = "he"
MAX_CHUNK_S = 25
SR = 16000


class Transcriber:
    def __init__(self, q: queue.Queue, output_path: str):
        self.q = q
        self.output_path = output_path

        self.ipa_model = WhisperModel(IPA_MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.text_model = WhisperModel(TEXT_MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.vad_model = load_silero_vad()

    def run(self, progress_start: int = 0):
        count = progress_start
        with open(self.output_path, "a", encoding="utf-8") as f:
            while True:
                item = self.q.get()
                if item is None:
                    break

                filename, audio, _ref_text = item
                full_text, full_ipa = self._transcribe(audio)

                f.write(f"{filename}\t{full_text}\t{full_ipa}\n")
                f.flush()

                count += 1
                print(f"[{count}] {filename} — done")

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        wav_tensor = torch.from_numpy(audio)
        timestamps = get_speech_timestamps(wav_tensor, self.vad_model, return_seconds=True, sampling_rate=SR)

        if not timestamps:
            return "", ""

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

        text_parts = []
        ipa_parts = []
        for chunk_start, chunk_end in merged:
            chunk = audio[chunk_start:chunk_end]

            text_segs, _ = self.text_model.transcribe(chunk, beam_size=5, language=LANGUAGE, temperature=0, condition_on_previous_text=False)
            ipa_segs, _ = self.ipa_model.transcribe(chunk, beam_size=5, language=LANGUAGE, temperature=0, condition_on_previous_text=False, no_speech_threshold=0.1)

            chunk_text = " ".join(s.text.strip() for s in text_segs)
            chunk_ipa = " ".join(s.text.strip() for s in ipa_segs)

            if chunk_text:
                text_parts.append(chunk_text)
            if chunk_ipa:
                ipa_parts.append(chunk_ipa)

        return " ".join(text_parts), " ".join(ipa_parts)
