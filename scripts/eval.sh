#!/usr/bin/env bash

set -x
set -e

model_path="bert"
task="WN18RR"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    model_path=$1
    shift
fi
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    task=$1
    shift
fi

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${task}"
fi

test_path="${DATA_DIR}/test.txt.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    test_path=$1
    shift
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
--task "${task}" \
--is-test \
--eval-model-path "${model_path}" \
--neighbor-weight "${neighbor_weight}" \
--rerank-n-hop "${rerank_n_hop}" \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${test_path}" "$@"
