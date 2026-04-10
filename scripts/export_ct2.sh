#!/bin/bash
MODEL_DIR=${1:-./whisper-heb-ipa}
OUTPUT_DIR=${2:-./whisper-heb-ipa-ct2}
export LD_LIBRARY_PATH=/home/yakov/Documents/spark-docs/CTranslate2/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

uv pip install 'ctranslate2>=4.6.0'
uv run ct2-transformers-converter \
    --model "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --quantization int8_float16