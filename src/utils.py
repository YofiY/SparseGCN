import torch
import numpy as np
import random

def generate_train_and_test_mask(num_nodes: int):
    train_mask_indices = random.sample(range(num_nodes), int(num_nodes*0.5))
    test_mask_indices = list(set(range(num_nodes)) - set(train_mask_indices))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_mask_indices] = 1
    test_mask[test_mask_indices] = 1
    return train_mask, test_mask

def get_edge_idx_from_adj_matrix(adj_matrix: np.ndarray):
    return torch.tensor(adj_matrix).nonzero().t().contiguous()
    
def get_num_edges_from_adj_matrix(adjacency_matrix):
    return int(np.count_nonzero(adjacency_matrix, axis = 1).sum() / 2)