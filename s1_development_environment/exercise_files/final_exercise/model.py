from torch import nn
import torch
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(784, 128)
        self.output = nn.Linear(128, 10)

        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
  
        x = F.relu(self.h1(x))
        x = self.output(x)

        return F.log_softmax(x, dim=1)