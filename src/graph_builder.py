import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import numpy as np

def build_graph_jet_graph(jet_features, label, k_neighbors = 16):
    mask = jet_features[:,0]>0
    valid_particles = jet_features[mask]

    if len(valid_particles) == 0:
        return None

    x = torch.tensor(valid_particles, dtype = torch.float)
    coords = x[:,1:3]
    num_nodes = coords.size(0)
    actual_k = min(k_neighbors, num_nodes)
    edge_index = knn_graph(coords, k=actual_k, loop = True)

    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


