"""
uv run python -m pipeline.run [--output output.tsv] [--max N] [--queue-size N]
"""
import argparse
import os
import queue

from datasets import load_dataset_builder
from tqdm import tqdm

from pipeline.downloader import Downloader, DATASET_NAME
from pipeline.preprocessor import Preprocessor
from pipeline.transcriber import Transcriber


def load_skip_ids(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    skip = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t", 1)
            if parts[0].strip():
                skip.add(parts[0].strip())
    return skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output.tsv")
    parser.add_argument("--max", type=int, default=None, dest="max_items")
    parser.add_argument("--queue-size", type=int, default=6)
    args = parser.parse_args()

    skip_ids = load_skip_ids(args.output)
    if skip_ids:
        print(f"Resuming: skipping {len(skip_ids)} already-done files")

    max_new = None
    if args.max_items is not None:
        max_new = max(0, args.max_items - len(skip_ids))

    download_q = queue.Queue(maxsize=args.queue_size)
    preprocess_q = queue.Queue(maxsize=args.queue_size)

    downloader = Downloader(download_q, skip_ids=skip_ids, max_items=max_new)
    preprocessor = Preprocessor(download_q, preprocess_q)
    transcriber = Transcriber(preprocess_q, output_path=args.output)

    downloader.start()
    preprocessor.start()
    total = load_dataset_builder(DATASET_NAME).info.splits["train"].num_examples
    transcriber.run(progress=tqdm(total=total, initial=len(skip_ids), unit="file"))
    downloader.join()
    preprocessor.join()


if __name__ == "__main__":
    main()
