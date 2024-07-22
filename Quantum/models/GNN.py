import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch

class GNN_regresion(torch.nn.Module):
    def __init__(self):
        super(GNN_regresion, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.gelu(x)
        
        x = global_mean_pool(x, data.batch)  # Global mean pooling
        
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    @classmethod
    def load(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model