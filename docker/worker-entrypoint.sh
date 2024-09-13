#!/bin/sh

## This script is supposed to run inside the container

if [ "$#" -ne 9 ]; then
  echo "Usage: $0 <worker_name> <model_path> <model_name> <kernel_name> (paged|static) <listen_ip> <listen_port> <coordinator_ip> <coordinator_port>"
  exit 1
fi

WORKER_NAME=$1
MODEL_DIR=$2
MODEL_NAME=$3
KERNEL_NAME=$4
PAGING_STRATEGY=$5
LISTEN_IP=$(hostname -I | awk '{print $1}')
LISTEN_PORT=$7
COORD_IP=$8
COORD_PORT=$9

_GLINTHAWK_OUTBOUND_LATENCY_=${_GLINTHAWK_OUTBOUND_LATENCY_:-0}

tc qdisc del dev eth0 root netem || true

if [ "${_GLINTHAWK_OUTBOUND_LATENCY_}" -ne 0 ]; then
  echo "Adding ${_GLINTHAWK_OUTBOUND_LATENCY_} ms of latency to all outbound traffic..."
  tc qdisc add dev eth0 root netem delay ${_GLINTHAWK_OUTBOUND_LATENCY_}ms
fi

/app/$WORKER_NAME $MODEL_DIR $MODEL_NAME $KERNEL_NAME $PAGING_STRATEGY $LISTEN_IP $LISTEN_PORT $COORD_IP $COORD_PORT
