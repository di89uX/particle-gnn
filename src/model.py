import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, global_mean_pool

class ParticleNetEdgeBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
        self.edge_conv = EdgeConv(nn=self.mlp, aggr='mean')

    def forward(self, x, edge_index):
        return self.edge_conv(x, edge_index)
class ParticleNet(nn.Module):
    def __init__(self, num_node_features=3, num_classes=2):
        super().__init__()
        self.conv1 = ParticleNetEdgeBlock(num_node_features, 64)
        self.conv2 = ParticleNetEdgeBlock(64, 128)
        self.conv3 = ParticleNetEdgeBlock(128, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x, edge_index, batch_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch_index)  # Global pooling
        out = self.classifier(x)
        return out