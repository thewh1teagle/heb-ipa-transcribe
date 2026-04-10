import multiprocessing
import queue
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
import numpy as np
from faster_whisper import WhisperModel

IPA_MODEL_ID = "thewh1teagle/abjad-he-ipa-ct2"
TEXT_MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEVICE = "cuda"
COMPUTE_TYPE = "int8"
LANGUAGE = "he"

_model: WhisperModel | None = None


def _init_worker(model_id: str):
    global _model
    _model = WhisperModel(model_id, device=DEVICE, compute_type=COMPUTE_TYPE)


def _transcribe_chunk(chunk: np.ndarray, is_ipa: bool) -> str:
    kwargs = dict(beam_size=5, language=LANGUAGE, temperature=0, condition_on_previous_text=False)
    if is_ipa:
        kwargs["no_speech_threshold"] = 0.1
    segs, _ = _model.transcribe(chunk, **kwargs)
    return " ".join(s.text.strip() for s in segs)


class Transcriber:
    def __init__(self, q: queue.Queue, output_path: str):
        self.q = q
        self.output_path = output_path

    def run(self, progress: tqdm):
        ctx = multiprocessing.get_context("spawn")
        with (
            ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_init_worker, initargs=(TEXT_MODEL_ID,)) as text_pool,
            ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_init_worker, initargs=(IPA_MODEL_ID,)) as ipa_pool,
            open(self.output_path, "a", encoding="utf-8") as f,
            progress,
        ):
            while True:
                item = self.q.get()
                if item is None:
                    break

                filename, chunks = item
                try:
                    for start_s, end_s, chunk in chunks:
                        text_future = text_pool.submit(_transcribe_chunk, chunk, False)
                        ipa_future = ipa_pool.submit(_transcribe_chunk, chunk, True)
                        text = text_future.result()
                        ipa = ipa_future.result()
                        if text or ipa:
                            f.write(f"{filename}\t{start_s:.3f}\t{end_s:.3f}\t{text}\t{ipa}\n")
                    f.flush()
                except Exception as e:
                    print(f"[transcriber] skipping {filename}: {e}")
                    progress.update(1)
                    continue

                progress.set_postfix_str(filename, refresh=False)
                progress.update(1)
