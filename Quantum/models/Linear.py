import torch
import torch.nn.functional as F

class linear_regresion(torch.nn.Module):
    def __init__(self, num_elements=23):
        super(linear_regresion, self).__init__()
        self.num_elements = num_elements
        self.fc = torch.nn.Linear(num_elements * num_elements, 1)

    def forward(self, data):
        x = data.edge_attr.reshape(-1, self.num_elements * self.num_elements)
        return self.fc(x)

class mlp(torch.nn.Module):
    def __init__(self, num_elements=23):
        super(mlp, self).__init__()
        self.num_elements = num_elements
        self.fc1 = torch.nn.Linear(num_elements * num_elements, 512)
        self.fc2 = torch.nn.Linear(512, 1)

    def forward(self, data):
        x = data.edge_attr.reshape(-1, self.num_elements * self.num_elements)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x