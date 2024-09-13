#!/bin/sh

## This script is supposed to run inside the container

if [ "$#" -ne 6 ]; then
  echo "Usage: $0 <worker_name> <model_dir> <listen_port> <coord_ip> <coord_port>"
  exit 1
fi

WORKER_NAME=$1
MODEL_DIR=$2
LISTEN_IP=$(hostname -I | awk '{print $1}')
LISTEN_PORT=$3
COORD_IP=$4
COORD_PORT=$5

_GLINTHAWK_OUTBOUND_LATENCY_=${_GLINTHAWK_OUTBOUND_LATENCY_:-0}

if [ "$_GLINTHAWK_OUTBOUND_LATENCY_" -ne 0 ]; then
  echo "Adding $_GLINTHAWK_OUTBOUND_LATENCY_ ms of latency to all outbound traffic..."
  tc qdisc add dev eth0 root netem delay ${_GLINTHAWK_OUTBOUND_LATENCY_}ms
fi

/app/$WORKER_NAME $MODEL_DIR $LISTEN_IP $LISTEN_PORT $COORD_IP $COORD_PORT
