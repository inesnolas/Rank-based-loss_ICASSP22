import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SingleLayerHypersphereConstraint(nn.Module):
    
    def __init__(self, configs ):
        super(SingleLayerHypersphereConstraint, self).__init__()
        self.linear = nn.Linear(configs["EMBEDDINGS_SIZE"], configs["output_EMBEDDINGS_SIZE"])
        self.optimizer = optim.SGD(self.parameters(), lr=configs["LR"])

    def forward(self, examples):  
        x = self.linear( examples)     
        x = torch.nn.functional.normalize(x, p=2)
        return x

