#!/bin/sh

if [ "$#" -ne 6 ]; then
  echo "Usage: $0 <model-dir> <prompt-dir> <listen-ip> <listen-port> <coord-ip> <coord-port>"
  exit 1
fi

MODEL_DIR=$1
PROMPT_DIR=$2
LISTEN_IP=$3
LISTEN_PORT=$4
COORD_IP=$5
COORD_PORT=$6

docker run -it --rm \
  --runtime=nvidia --gpus all \
  --network host --no-healthcheck --read-only --ulimit nofile=65535:65535 \
  --mount type=bind,src="$1",dst="/app/model/",readonly \
  --mount type=bind,src="$2",dst="/app/prompts/" \
  glinthawk-worker /app/model/ $LISTEN_IP $LISTEN_PORT $COORD_IP $COORD_PORT
