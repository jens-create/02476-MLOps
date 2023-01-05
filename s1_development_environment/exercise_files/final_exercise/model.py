import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning import LightningModule


class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(784, 128)
        self.output = nn.Linear(128, 10)
        self.criterium = nn.NLLLoss()

    def forward(self, x):
        """Forward pass through the network, returns the output logits"""
        print(x.shape)
        x = F.relu(self.h1(x))
        x = self.output(x)

        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
