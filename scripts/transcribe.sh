#!/bin/bash
export LD_LIBRARY_PATH=/home/yakov/Documents/spark-docs/CTranslate2/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
uv run python -m src.run "$@"
uv run hf repo create knesset-ipa --type dataset || true
uv run hf upload knesset-ipa ./output.tsv