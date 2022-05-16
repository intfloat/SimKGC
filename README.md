## SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models

Official code repository for ACL 2022 paper 
"[SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models](https://aclanthology.org/2022.acl-long.295.pdf)".

The paper is available at [https://aclanthology.org/2022.acl-long.295.pdf](https://aclanthology.org/2022.acl-long.295.pdf).

In this paper,
we identify that one key issue for text-based knowledge graph completion is efficient contrastive learning.
By combining large number of negatives and hardness-aware InfoNCE loss,
SimKGC can substantially outperform existing methods on popular benchmark datasets.

## Requirements
* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15

All experiments are run with 4 V100(32GB) GPUs.

## How to Run

It involves 3 steps: dataset preprocessing, model training, and model evaluation.

We also provide the predictions from our models in [predictions](predictions/) directory.

For WN18RR and FB15k237 datasets, we use files from [KG-BERT](https://github.com/yao8839836/kg-bert).

### WN18RR dataset

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh WN18RR
```

Step 2, training the model and (optionally) specify the output directory (< 3 hours)
```
OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR
```

Feel free to change the output directory to any path you think appropriate.

### FB15k-237 dataset

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh FB15k237
```

Step 2, training the model and (optionally) specify the output directory (< 3 hours)
```
OUTPUT_DIR=./checkpoint/fb15k237/ bash scripts/train_fb.sh
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/fb15k237/model_last.mdl FB15k237
```

### Wikidata5M transductive dataset

Step 0, download the dataset. 
We provide a script to download the [Wikidata5M dataset](https://deepgraphlearning.github.io/project/wikidata5m) from its official website.
This will download data for both transductive and inductive settings.
```
bash ./scripts/download_wikidata5m.sh
```

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh wiki5m_trans
```

Step 2, training the model and (optionally) specify the output directory (about 12 hours)
```
OUTPUT_DIR=./checkpoint/wiki5m_trans/ bash scripts/train_wiki.sh wiki5m_trans
```

Step 3, evaluate a trained model (it takes about 1 hour due to the large number of entities)
```
bash scripts/eval_wiki5m_trans.sh ./checkpoint/wiki5m_trans/model_last.mdl
```

### Wikidata5M inductive dataset

Make sure you have run `scripts/download_wikidata5m.sh` to download Wikidata5M dataset.

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh wiki5m_ind
```

Step 2, training the model and (optionally) specify the output directory (about 11 hours)
```
OUTPUT_DIR=./checkpoint/wiki5m_ind/ bash scripts/train_wiki.sh wiki5m_ind
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/wiki5m_ind/model_last.mdl wiki5m_ind
```

## Troubleshooting

1. I encountered "CUDA out of memory" when running the code.

We run experiments with 4 V100(32GB) GPUs, please reduce the batch size if you don't have enough resources. 
Be aware that smaller batch size will hurt the performance for contrastive training. 

2. Does this codebase support distributed data parallel(DDP) training?

No. Some input masks require access to batch data on all GPUs, 
so currently it only supports data parallel training for ease of implementation.

## Citation

If you find our paper or code repository helpful, please consider citing as follows:

```
@inproceedings{wang-etal-2022-simkgc,
    title = "{S}im{KGC}: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models",
    author = "Wang, Liang  and
      Zhao, Wei  and
      Wei, Zhuoyu  and
      Liu, Jingming",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.295",
    pages = "4281--4294",
}
```
