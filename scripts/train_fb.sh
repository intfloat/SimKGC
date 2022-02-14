#!/usr/bin/env bash

set -x
set -e

device_id=0
model_name="test"
task="FB15k237"
if [[ $# -ge 1 && ! "$1" =~ "--"* ]]; then
    device_id=$1
    shift
fi
if [[ $# -ge 1 && ! "$1" =~ "--"* ]]; then
    model_name=$1
    shift
fi
if [[ $# -ge 1 && ! "$1" =~ "--"* ]]; then
    task=$1
    shift
fi

MODEL_SAVE_DIR="./checkpoint/${model_name}-`date +%F-%H%M.%S`/"
LOG="${MODEL_SAVE_DIR}/run.log"
mkdir -p ${MODEL_SAVE_DIR}

CUDA_VISIBLE_DEVICES=${device_id} nohup python3 -u main.py \
--model-dir ${MODEL_SAVE_DIR} \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 1e-5 \
--use-link-graph \
--train-path ./data/${task}/train.txt.json \
--valid-path ./data/${task}/valid.txt.json \
--task ${task} \
--batch-size 1024 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--pre-batch 2 \
--epochs 10 \
--workers 4 \
--max-to-keep 5 "$@" > ${LOG} 2>&1 &
