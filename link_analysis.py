import json
import argparse

from collections import defaultdict
from typing import List
from triplet import LinkGraph

parser = argparse.ArgumentParser(description='link analysis')
parser.add_argument('--train-path', default='', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='', type=str, metavar='N',
                    help='path to valid data')
args = parser.parse_args()


def _load(path: str) -> List[dict]:
    examples = json.load(open(path, 'r', encoding='utf-8'))
    print('Load {} examples from {}'.format(len(examples), path))
    return examples


def _load_train_graph() -> dict:
    examples = _load(args.train_path)
    # id -> set(id)
    id2nodes = defaultdict(set)
    for ex in examples:
        head_id, tail_id = ex['head_id'], ex['tail_id']
        id2nodes[head_id].add(tail_id)
        id2nodes[tail_id].add(head_id)

    return id2nodes


def _distance_analysis():
    graph = LinkGraph(args.train_path)
    examples = _load(args.valid_path)
    len2cnt = {}
    res = []
    for ex in examples:
        src, dst = ex['head_id'], ex['tail_id']
        path_len, visit_cnt = graph.shortest_path(src=src, dst=dst)
        len2cnt[path_len] = len2cnt.get(path_len, 0) + 1
        res.append({'src': ex['head'], 'dst': ex['tail'], 'path_len': path_len, 'visit_cnt': visit_cnt})

    for length, cnt in sorted(len2cnt.items(), key=lambda tp: tp[0]):
        print('Path length={}, cnt={}, percentage={}%'
              .format(length, cnt, round(100 * cnt / len(examples), 4)))

    res = sorted(res, key=lambda d: d['path_len'])
    json.dump(res, open('./log_link_analysis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # python3 link_analysis.py --train-path ./data/FB15k237/train.txt.json --valid-path ./data/FB15k237/valid.txt.json
    _distance_analysis()
