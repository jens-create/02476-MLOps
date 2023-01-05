from pytorch_lightning import Trainer

import click
import torch
from model import MyAwesomeModel
from torch import nn

from data import mnist


model = MyAwesomeModel()  # this is our LightningModule
trainloader, testloader = mnist()  # data
trainer = Trainer(max_epochs=10)
trainer.fit(model, trainloader, testloader)
