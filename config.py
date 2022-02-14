import os
import random
import torch
import argparse
import warnings

import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='example usage for various pretrained models')
parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, metavar='N',
                    help='path to pretrained model')
parser.add_argument('--arch', default='bert', type=str, metavar='N',
                    help='model architecture')
parser.add_argument('--task', default='wn18rr', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--train-path', default='', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--model-dir', default='', type=str, metavar='N',
                    help='path to model dir')
parser.add_argument('--warmup', default=400, type=int, metavar='N',
                    help='warmup steps')
parser.add_argument('--freeze-embedding', action='store_true',
                    help='freeze embedding or not')
parser.add_argument('--freeze-n-layers', default=0, type=int, metavar='N',
                    help='freeze bottom n layers')
parser.add_argument('--max-to-keep', default=5, type=int, metavar='N',
                    help='max number of checkpoints to keep')
parser.add_argument('--grad-clip', default=10.0, type=float, metavar='N',
                    help='gradient clipping')
parser.add_argument('--pooling', default='cls', type=str, metavar='N',
                    help='bert pooling')
parser.add_argument('--dropout', default=0.1, type=float, metavar='N',
                    help='dropout on final linear layer')
parser.add_argument('--use-amp', action='store_true',
                    help='Use amp if available')
parser.add_argument('--t', default=0.05, type=float,
                    help='temperature parameter')
parser.add_argument('--use-link-graph', action='store_true',
                    help='use neighbors from link graph as context')
parser.add_argument('--eval-every-n-step', default=10000, type=int,
                    help='evaluate every n steps')
parser.add_argument('--pre-batch', default=0, type=int,
                    help='number of pre-batch used for negatives')
parser.add_argument('--pre-batch-weight', default=0.5, type=float,
                    help='the logit weight for logits from pre-batch negatives')
parser.add_argument('--additive-margin', default=0.0, type=float, metavar='N',
                    help='additive margin for InfoNCE loss function')
parser.add_argument('--finetune-t', action='store_true',
                    help='make temperature as a trainable parameter or not')
parser.add_argument('--max-num-tokens', default=50, type=int,
                    help='maximum number of tokens')
parser.add_argument('--use-self-negative', action='store_true',
                    help='use head entity as negative')

# following options are only used for evaluation
parser.add_argument('--is-test', action='store_true',
                    help='is in test mode or not')
parser.add_argument('--rerank-n-hop', default=2, type=int,
                    help='use n-hops node for re-ranking entities, only used during evaluation')

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# only used for evaluation
parser.add_argument('--neighbor-weight', default=0.0, type=float,
                    help='weight for re-ranking entities')

args = parser.parse_args()

assert not args.train_path or os.path.exists(args.train_path)
assert os.path.exists(args.model_dir)
assert args.pooling in ['cls', 'mean', 'max', 'mean_first_last']
assert args.task.lower() in ['wn18rr', 'fb15k237', 'conceptnet', 'wiki5m_ind', 'wiki5m_trans']
assert args.lr_scheduler in ['linear', 'cosine']

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

args.distributed = args.world_size > 1 or args.multiprocessing_distributed

try:
    if args.use_amp:
        import torch.cuda.amp
except Exception:
    args.use_amp = False
    warnings.warn('AMP training is not available, set use_amp=False')
