#!/bin/bash
export LD_LIBRARY_PATH=/home/yakov/Documents/spark-docs/CTranslate2/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
uv run python -m src.run "$@"
