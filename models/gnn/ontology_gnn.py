# models/gnn/ontology_gnn.py
import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv

class OntologyGNN(torch.nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=256, num_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, out_dim))
        self.act = ReLU()

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = self.act(conv(x, edge_index))
        return self.layers[-1](x, edge_index)
