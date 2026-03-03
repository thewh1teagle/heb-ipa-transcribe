"""
uv run python -m pipeline.run [--output output.tsv] [--max N] [--queue-size N]
"""
import argparse
import os
import queue

from datasets import load_dataset_builder
from tqdm import tqdm

from pipeline.downloader import Downloader, DATASET_NAME
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

    builder = load_dataset_builder(DATASET_NAME)
    total = builder.info.splits["train"].num_examples

    max_new = None
    if args.max_items is not None:
        max_new = max(0, args.max_items - len(skip_ids))

    progress_total = max_new if max_new is not None else total - len(skip_ids)

    q = queue.Queue(maxsize=args.queue_size)
    downloader = Downloader(q, skip_ids=skip_ids, max_items=max_new)
    transcriber = Transcriber(q, output_path=args.output)

    downloader.start()
    transcriber.run(progress=tqdm(total=progress_total, initial=0, unit="file"))
    downloader.join()


if __name__ == "__main__":
    main()
