"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
import torchvision.transforms.functional as F


def show(imgs):
    plt.rcParams["savefig.bbox"] = "tight"
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.savefig("batch.png")


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:

        print(path_to_folder)
        self.root_dir = path_to_folder

        idx = 0
        dataset = []
        for dir in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, dir)):
                for image in os.listdir(os.path.join(self.root_dir, dir)):
                    dataset.append(
                        {
                            "path": os.path.join(self.root_dir, dir, image),
                            "person": dir,
                            "idx": idx,
                        }
                    )
                    idx += 1
        self.dataset = dataset
        self.length = len(dataset)

        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out

        img_name = self.dataset[index]["path"]  # os.path.join(self.root_dir, se)
        img = Image.open(img_name)

        return self.transform(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="~/Downloads/lfw", type=str)
    parser.add_argument("-batch_size", default=512, type=int)
    parser.add_argument("-num_workers", default=0, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=5, type=int)

    args = parser.parse_args()

    lfw_trans = transforms.Compose(
        [
            transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
            transforms.ToTensor(),
            transforms.CenterCrop(10),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    if args.visualize_batch:
        # TODO: visualize a batch of images
        images = next(iter(dataloader))
        show(make_grid(images))

    if args.get_timing:
        # lets do some repetitions
        res = []
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)

        res = np.array(res)
        print(f"Timing: {np.mean(res)}+-{np.std(res)}")


# simple transformation
# Timing: 11.45329647064209+-0.3680623753803977 - 1
# Timing: 16.211685562133788+-0.47402063662394  - 2
# Timing: 22.143654346466064+-0.42288135790120485 - 3
# Timing: 34.76241321563721+-0.2747690737688849 -5


# Timing: 3.625906467437744+-0.1399087847642243 - 0


# it is slower to use num_workers > 0
