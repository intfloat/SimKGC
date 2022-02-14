import glob
import os
import json
import torch
import numpy as np

from typing import List
from collections import Counter
from dataclasses import dataclass, asdict

from config import args
from triplet import EntityDict
from dict_hub import get_all_triplet_dict, get_entity_dict
from logger_config import logger

data_dir = './data/wiki5m_trans/'
eval_forward_path = 'checkpoint/0823_wiki_trans_self_negative-2021-08-23-2246.33/' \
                    'eval_test.txt.json_forward_checkpoint_epoch0.mdl.json'
eval_backward_path = 'checkpoint/0823_wiki_trans_self_negative-2021-08-23-2246.33/' \
                     'eval_test.txt.json_backward_checkpoint_epoch0.mdl.json'

# assert os.path.exists(data_dir)
# assert os.path.exists(eval_forward_path)
# assert os.path.exists(eval_backward_path)

entity_dict = None


@dataclass
class EvalInfo:
    head: str
    head_desc: str
    relation: str
    tail: str
    pred_tail: str
    pred_tail_desc: str
    pred_score: float
    human_label: str = ''


wiki5m_id2text = {}


def _truncate(text: str, max_len: int):
    return ' '.join(text.split()[:max_len])


def _load_wiki5m_id2text(path: str, max_len: int = 60):
    global wiki5m_id2text
    for line in open(path, 'r', encoding='utf-8'):
        fs = line.strip().split('\t')
        assert len(fs) >= 2, 'Invalid line: {}'.format(line.strip())
        ent_id, ent_text = fs[0], ' '.join(fs[1:])
        wiki5m_id2text[ent_id] = _truncate(ent_text, max_len)

    logger.info('Load {} entity texts from {}'.format(len(wiki5m_id2text), path))


def _random_select(path: str, cnt: int) -> List[EvalInfo]:
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(entity_dict_dir=data_dir)

    examples = json.load(open(path, 'r', encoding='utf-8'))
    examples = [ex for ex in examples if not ex['correct']]
    np.random.shuffle(examples)
    examples = examples[:cnt]

    for ex in examples:
        ex['head_desc'] = wiki5m_id2text[ex['head']]
        ex['pred_tail_desc'] = wiki5m_id2text[ex['pred_tail']]
        ex['head'] = entity_dict.get_entity_by_id(ex['head']).entity
        ex['tail'] = entity_dict.get_entity_by_id(ex['tail']).entity
        ex['pred_tail'] = entity_dict.get_entity_by_id(ex['pred_tail']).entity

        del ex['correct'], ex['topk_score_info']

    examples = [EvalInfo(**ex) for ex in examples]
    return examples


def _get_human_eval_data(cnt=100):
    _load_wiki5m_id2text(path=os.path.join(data_dir, 'wikidata5m_text.txt'))

    examples = _random_select(eval_forward_path, cnt // 2)
    examples += _random_select(eval_backward_path, cnt // 2)
    np.random.shuffle(examples)
    logger.info('Get {} examples'.format(len(examples)))

    out_path = os.path.join(os.path.dirname(eval_backward_path), 'human_eval.json')
    json.dump([asdict(ex) for ex in examples], open(out_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    logger.info('Done')


def _human_eval_stat(path='./data/wiki5m_trans_human_eval.json'):
    examples = json.load(open(path, 'r', encoding='utf-8'))
    forward_cnt, backward_cnt, cnt = Counter(), Counter(), Counter()
    for ex in examples:
        cnt[ex['human_label']] += 1
        if ex['relation'].startswith('inverse'):
            backward_cnt[ex['human_label']] += 1
        else:
            forward_cnt[ex['human_label']] += 1

    print('total count: ', cnt)
    print('forward count: ', forward_cnt)
    print('backward count:', backward_cnt)


def _inverse_rel(r):
    return 'inverse {}'.format(r)


def _classify_relations(data_dir):
    if os.path.exists('{}/rel_type.json'.format(data_dir)):
        logger.info('relation type already exists')
        return

    # relation -> [tot_entity_cnt, num_distinct]
    from collections import defaultdict, Counter
    r2tail_cnt = defaultdict(lambda: [0, 0])
    r2head_cnt = defaultdict(lambda: [0, 0])

    hr_cnt, rt_cnt = Counter(), Counter()
    for path in glob.glob('{}/*.txt.json'.format(data_dir)):
        logger.info('Load data from {}'.format(path))
        examples = json.load(open(path, 'r', encoding='utf-8'))
        for ex in examples:
            head, relation, tail = ex['head_id'], ex['relation'], ex['tail_id']
            hr_cnt[(head, relation)] += 1
            rt_cnt[(tail, relation)] += 1
    logger.info('Finish data loading')

    # 1-1, 1-n: average number of tails given <h, r>
    # n-1: average number of heads given <r, t>
    # n-n: both 1-n and n-1
    for (head, relation), val in hr_cnt.items():
        cnt = r2tail_cnt[relation]
        cnt[0] += val
        cnt[1] += 1
    for (tail, relation), val in rt_cnt.items():
        cnt = r2head_cnt[relation]
        cnt[0] += val
        cnt[1] += 1

    r2first, r2second = {}, {}
    for relation, val in r2tail_cnt.items():
        r2first[relation] = val[0] / val[1]
    for relation, val in r2head_cnt.items():
        r2second[relation] = val[0] / val[1]

    rel_stat = {r: (r2second[r], r2first[r]) for r in r2first}
    json.dump(rel_stat, open('{}/rel_stat.json'.format(data_dir), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    rel_type = {}
    for rel, (l, r) in rel_stat.items():
        if l < 1.5 and r < 1.5:
            rel_type[rel] = '1-1'
            rel_type[_inverse_rel(rel)] = '1-1'
        elif l < 1.5:
            rel_type[rel] = '1-n'
            rel_type[_inverse_rel(rel)] = 'n-1'
        elif r < 1.5:
            rel_type[rel] = 'n-1'
            rel_type[_inverse_rel(rel)] = '1-n'
        else:
            rel_type[rel] = 'n-n'
            rel_type[_inverse_rel(rel)] = 'n-n'
    json.dump(rel_type, open('{}/rel_type.json'.format(data_dir), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    logger.info('Done')


def _calc_metrics_by_relation_type(data_dir):
    from collections import defaultdict
    rel_type = json.load(open('{}/rel_type.json'.format(data_dir), 'r', encoding='utf-8'))
    from config import args
    pred_path_list = glob.glob('{}/eval_test*.json'.format(args.model_dir))
    type2rank = defaultdict(list)

    examples = []
    for path in pred_path_list:
        examples.extend(json.load(open(path, 'r', encoding='utf-8')))
    for ex in examples:
        type2rank[rel_type[ex['relation']]].append(ex['rank'])
    for t, ranks in type2rank.items():
        logger.info('type={}, mrr={}'.format(t, np.average([1 / rank for rank in ranks])))


def _prepare_vis_data(data_dir='./data/wiki5m_trans/'):
    if os.path.exists('./tail_to_20heads.json'):
        return

    # P31: instance of
    examples = []
    for path in glob.glob('{}/*.txt.json'.format(data_dir)):
        handle = open(path, 'r', encoding='utf-8')
        examples.extend([ex for ex in json.load(handle) if ex['relation'] == 'instance of'])
    logger.info('Load {} examples'.format(len(examples)))

    tail_cnt = Counter()
    for ex in examples:
        tail_cnt[ex['tail_id']] += 1
    freq_tail = set([t[0] for t in tail_cnt.most_common(8)])
    logger.info('frequent tail entity: {}'.format(freq_tail))

    from collections import defaultdict
    tail_to_head = defaultdict(set)
    for ex in examples:
        if ex['tail_id'] not in freq_tail:
            continue
        tail_to_head[ex['tail_id']].add(ex['head_id'])
    tail_to_20heads = {}
    for t in tail_to_head:
        all_heads = list(tail_to_head[t])
        np.random.shuffle(all_heads)
        tail_to_20heads[t] = all_heads[:50]
    json.dump(tail_to_20heads, open('./tail_to_20heads.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    logger.info('Done')


# copy & paste from eval_wiki5m_trans.py
entity_dict = get_entity_dict()
SHARD_SIZE = 1000000


def _get_shard_path(shard_id=0):
    return '{}_shard_{}'.format(args.model_dir, shard_id)


def _load_entity_embeddings():
    assert os.path.exists(_get_shard_path())

    shard_tensors = []
    for start in range(0, len(entity_dict), SHARD_SIZE):
        shard_id = start // SHARD_SIZE
        shard_path = _get_shard_path(shard_id=shard_id)
        shard_entity_tensor = torch.load(shard_path, map_location=lambda storage, loc: storage)
        logger.info('Load {} entity embeddings from {}'.format(shard_entity_tensor.size(0), shard_path))
        shard_tensors.append(shard_entity_tensor)

    entity_tensor = torch.cat(shard_tensors, dim=0)
    logger.info('{} entity embeddings in total'.format(entity_tensor.size(0)))
    logger.info('tensor device: {}'.format(entity_tensor.device))
    assert entity_tensor.size(0) == len(entity_dict.entity_exs)
    return entity_tensor


def _get_vectors():
    embeddings = _load_entity_embeddings()
    tail_to_heads = json.load(open('./tail_to_20heads.json', 'r', encoding='utf-8'))
    examples = []
    for t, heads in tail_to_heads.items():
        for h in heads:
            vector = embeddings[entity_dict.entity_to_idx(h)].tolist()
            examples.append({'tail_id': t,
                             'head_id': h,
                             'head_vector': ','.join([str(v) for v in vector])})
    json.dump(examples, open('./vectors.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # python3 analyze_data.py --task wiki5m_trans --valid-path ./data/wiki5m_trans/valid.txt.json
    # --train-path ./data/wiki5m_trans/train.txt.json --model-dir /path/to/model
    _prepare_vis_data()
    _get_vectors()

    # _get_human_eval_data()
    # _human_eval_stat()
    # from config import args
    # data_dir = './data/{}/'.format(args.task)
    # logger.info('model_dir={}'.format(args.model_dir))
    # _classify_relations(data_dir=data_dir)
    # _calc_metrics_by_relation_type(data_dir=data_dir)
    pass
