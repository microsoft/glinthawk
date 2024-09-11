#!/bin/bash

if [ $# -lt 4 ] || [ $# -gt 7 ];  then
  echo "Usage: $0 <model-dir> <model-name> <context_type=(paged|static)> <stage=(all|all_no_cls|pre|att|post|cls)> [<token_pos=2048> <duration=30> <LOG_DIR=/output_logs/>]"
  exit 1
fi

MODEL_DIR=$1
MODEL_NAME=$2
CTX_TYPE=$3
STAGE=$4
TOKEN_POS=${5:-"2047"}
DURATION=${6:-"30"}
LOG_DIR=${7:-"/output_logs/"}

batch_size_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 26 28 30 32 36 40 44 48 52 56 60 64 72 80 88 96 104 112 120 128 144 160 176 192 208 224 240 256 288 320 352 384 416 448 480 512 576 640 704 768 832 896 960 1024 1152 1280 1408 1536 1664 1792 1920 2048 2304 2560 2816 3072 3328 3584 3840 4096)

for batch_size in "${batch_size_list[@]}"
do
    echo "Batch size: $batch_size"
    /app/profile-stage $MODEL_DIR $MODEL_NAME $CTX_TYPE $STAGE $batch_size $TOKEN_POS $DURATION "${LOG_DIR}/${MODEL_NAME}_${STAGE}_${CTX_TYPE}_${TOKEN_POS}_${DURATION}_${batch_size}.csv"
done