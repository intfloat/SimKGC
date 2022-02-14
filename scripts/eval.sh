#!/usr/bin/env bash

set -x
set -e

device_id=0
model_path="bert"
task="WN18RR"
if [ $# -ge 1 ]; then
    device_id=$1
fi
if [ $# -ge 2 ]; then
    model_path=$2
fi
if [ $# -ge 3 ]; then
    task=$3
fi
test_path="./data/${task}/test.txt.json"
if [ $# -ge 4 ]; then
    test_path=$4
fi

neighbor_weight=0.05
rerank_n_hop=2
if [ "${task}" = "WN18RR" ]; then
# WordNet is a sparse graph, use more neighbors for re-rank
  rerank_n_hop=5
fi
if [ "${task}" = "wiki5m_ind" ]; then
# for inductive setting of wiki5m, test nodes never appear in the training set
  neighbor_weight=0.0
fi

python3 -u evaluate.py \
--gpu "${device_id}" \
--task "${task}" \
--is-test \
--model-dir "${model_path}" \
--neighbor-weight "${neighbor_weight}" \
--rerank-n-hop "${rerank_n_hop}" \
--train-path "./data/${task}/train.txt.json" \
--valid-path "${test_path}"
