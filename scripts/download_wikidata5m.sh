#!/usr/bin/env bash

set -x
set -e

REPO_DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
BASE_DIR="${REPO_DIR}/data/wikidata5m"
mkdir -p ${BASE_DIR}

wget -O "${BASE_DIR}/wikidata5m_text.txt.gz" https://huggingface.co/datasets/intfloat/wikidata5m/resolve/main/wikidata5m_text.txt.gz
wget -O "${BASE_DIR}/wikidata5m_transductive.tar.gz" https://huggingface.co/datasets/intfloat/wikidata5m/resolve/main/wikidata5m_transductive.tar.gz
wget -O "${BASE_DIR}/wikidata5m_inductive.tar.gz" https://huggingface.co/datasets/intfloat/wikidata5m/resolve/main/wikidata5m_inductive.tar.gz
wget -O "${BASE_DIR}/wikidata5m_alias.tar.gz" https://huggingface.co/datasets/intfloat/wikidata5m/resolve/main/wikidata5m_alias.tar.gz

cd ${BASE_DIR}
tar xvfz "wikidata5m_transductive.tar.gz"
tar xvfz "wikidata5m_inductive.tar.gz"
tar xvfz "wikidata5m_alias.tar.gz"
gunzip -k "wikidata5m_text.txt.gz"
cd ../../

TRANS_DIR="${REPO_DIR}/data/wiki5m_trans/"
mkdir -p ${TRANS_DIR}
ln -s "${BASE_DIR}/wikidata5m_relation.txt" ${TRANS_DIR}
ln -s "${BASE_DIR}/wikidata5m_text.txt" ${TRANS_DIR}
ln -s "${BASE_DIR}/wikidata5m_entity.txt" ${TRANS_DIR}
ln -s "${BASE_DIR}/wikidata5m_transductive_train.txt" "${TRANS_DIR}/train.txt"
ln -s "${BASE_DIR}/wikidata5m_transductive_valid.txt" "${TRANS_DIR}/valid.txt"
ln -s "${BASE_DIR}/wikidata5m_transductive_test.txt" "${TRANS_DIR}/test.txt"

IND_DIR="${REPO_DIR}/data/wiki5m_ind/"
mkdir -p ${IND_DIR}
ln -s "${BASE_DIR}/wikidata5m_relation.txt" ${IND_DIR}
ln -s "${BASE_DIR}/wikidata5m_text.txt" ${IND_DIR}
ln -s "${BASE_DIR}/wikidata5m_entity.txt" ${IND_DIR}
ln -s "${BASE_DIR}/wikidata5m_inductive_train.txt" "${IND_DIR}/train.txt"
ln -s "${BASE_DIR}/wikidata5m_inductive_valid.txt" "${IND_DIR}/valid.txt"
ln -s "${BASE_DIR}/wikidata5m_inductive_test.txt" "${IND_DIR}/test.txt"

echo "Done"
