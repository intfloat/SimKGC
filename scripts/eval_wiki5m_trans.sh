#!/usr/bin/env bash

set -x
set -e

device_id=0
model_path="bert"
task="wiki5m_trans"
if [ $# -ge 1 ]; then
    device_id=$1
fi
if [ $# -ge 2 ]; then
    model_path=$2
fi
test_path="./data/${task}/test.txt.json"
if [ $# -ge 3 ]; then
    test_path=$3
fi

neighbor_weight=0.05

CUDA_VISIBLE_DEVICES=${device_id} python3 -u eval_wiki5m_trans.py \
--gpu 0 \
--task "${task}" \
--is-test \
--model-dir "${model_path}" \
--neighbor-weight "${neighbor_weight}" \
--train-path "./data/${task}/train.txt.json" \
--valid-path "${test_path}"
