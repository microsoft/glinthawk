#!/bin/bash

if [ $# -lt 2 ] || [ $# -gt 4 ];  then
  echo "Usage: $0 dim2 dim3 [<duration=30> <LOG_DIR=/output_logs/>]"
  exit 1
fi

DIM1=$1
DIM2=$2
DURATION=${3:-"30"}
LOG_DIR=${4:-"/output_logs/"}

batch_size_list=(1 2 3 4 6 8 12 16 24 32 48 64 96 128 192 256 384 512 768 1024 1536 2048 3072 4096)

for batch_size in "${batch_size_list[@]}"
do
    echo "Batch size: $batch_size"
    /app/profile-gemm $batch_size $DIM1 $DIM2 $DURATION "${LOG_DIR}/gemm_${batch_size}_${DIM1}_${DIM2}_${DURATION}.csv"
done