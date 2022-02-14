#!/usr/bin/env bash

set -x
set -e

task="WN18RR"
#task="FB15k237"
if [[ $# -ge 1 ]]; then
    task=$1
    shift
fi

python3 -u preprocess.py \
--task ${task} \
--train-path "./data/${task}/train.txt" \
--valid-path "./data/${task}/valid.txt" \
--test-path "./data/${task}/test.txt" &
