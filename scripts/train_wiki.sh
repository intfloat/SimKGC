#!/usr/bin/env bash

set -x
set -e

device_id=0
model_name="test"
task="wiki5m_ind"
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
--lr 3e-5 \
--train-path ./data/${task}/train.txt.json \
--valid-path ./data/${task}/valid.txt.json \
--task ${task} \
--batch-size 1024 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--pre-batch 1 \
--epochs 1 \
--workers 4 \
--max-to-keep 10 "$@" > ${LOG} 2>&1 &
