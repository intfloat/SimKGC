import datasets

from datasets import load_dataset


def process(batch):
    num_exs = len(batch['head_id'])
    batch['head_id'] += batch['tail_id'][:num_exs]
    batch['tail_id'] += batch['head_id'][:num_exs]
    batch['relation'] += ['inverse {}'.format(batch['relation'][i]) for i in range(num_exs)]
    return batch


if __name__ == '__main__':
    dataset = load_dataset('json', data_files=['./data/WN18RR/valid.txt.json'], split=datasets.Split.TRAIN)
    print(dataset[0])
    dataset = dataset.remove_columns(column_names=['head', 'tail'])
    print(dataset[0])
    dataset = dataset.map(process, batched=True)
    print(dataset[0])
    print(dataset[1])
