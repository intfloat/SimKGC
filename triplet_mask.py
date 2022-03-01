import torch

from typing import List

from config import args
from dict_hub import get_train_triplet_dict, get_entity_dict, EntityDict, TripletDict

entity_dict: EntityDict = get_entity_dict()
train_triplet_dict: TripletDict = get_train_triplet_dict() if not args.is_test else None


def construct_mask(row_exs: List, col_exs: List = None) -> torch.tensor:
    positive_on_diagonal = col_exs is None
    num_row = len(row_exs)
    col_exs = row_exs if col_exs is None else col_exs
    num_col = len(col_exs)

    # exact match
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_exs])
    col_entity_ids = row_entity_ids if positive_on_diagonal else \
        torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_exs])
    # num_row x num_col
    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
    if positive_on_diagonal:
        triplet_mask.fill_diagonal_(True)

    # mask out other possible neighbors
    for i in range(num_row):
        head_id, relation = row_exs[i].head_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        # exact match is enough, no further check needed
        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j and positive_on_diagonal:
                continue
            tail_id = col_exs[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask


def construct_self_negative_mask(exs: List) -> torch.tensor:
    mask = torch.ones(len(exs))
    for idx, ex in enumerate(exs):
        head_id, relation = ex.head_id, ex.relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        if head_id in neighbor_ids:
            mask[idx] = 0
    return mask.bool()
