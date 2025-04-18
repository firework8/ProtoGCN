#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

CONFIG=$1
GPUS=$2

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
