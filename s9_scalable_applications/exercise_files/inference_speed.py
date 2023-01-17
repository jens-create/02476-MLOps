import time
from torchvision import models
import torch

n_reps = 1

m1 = models.efficientnet_b5("IMAGENET1K_V1")
# m2 = models.resnet50('IMAGENET1K_V1')
# m3 = models.swin_b('IMAGENET1K_V1')

input = torch.randn(100, 3, 256, 256)

for i, m in enumerate([m1]):
    tic = time.time()
    for _ in range(n_reps):
        _ = m(input)
    toc = time.time()
    print(f"Model {i} took: {(toc - tic) / n_reps}")
