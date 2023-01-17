import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


def mnist():
    # exchange with the corrupted mnist dataset
    train_data_index = [0, 1, 2, 3, 4]
    X_train_data = torch.tensor([])
    y_train_data = torch.tensor([])
    for t in train_data_index:
        d = np.load(
            "/Users/jenspt/Desktop/git/02476-MLOps/data/corruptmnist/train_"
            + str(t)
            + ".npz"
        )
        X_d = torch.from_numpy(d["images"])
        y_d = torch.from_numpy(d["labels"])
        X_train_data = torch.cat((X_train_data, X_d))
        y_train_data = torch.cat((y_train_data, y_d))

    X_train_data = X_train_data.to(torch.float32)
    y_train_data = y_train_data.to(torch.int64)

    X_train_data.resize_(X_train_data.size()[0], 784)

    # Define a transform to normalize the data
    # transform = transforms.Compose([transforms.ToTensor(),
    #                            transforms.Normalize((0.5,), (0.5,)),
    #                          ])

    # Create dataset from several tensors with matching first dimension
    # Samples will be drawn from the first dimension (rows)
    traindataset = TensorDataset(X_train_data, y_train_data)
    # Create a data loader from the dataset
    # Type of sampling and batch size are specified at this step
    trainloader = DataLoader(traindataset, batch_size=32, shuffle=True)

    test_d = np.load("/Users/jenspt/Desktop/git/02476-MLOps/data/corruptmnist/test.npz")
    X_test = torch.from_numpy(test_d["images"])
    X_test.resize_(X_test.size()[0], 784)
    y_test = torch.from_numpy(test_d["labels"])

    X_test = X_test.to(torch.float32)
    y_test = y_test.to(torch.int64)

    testdataset = TensorDataset(X_test, y_test)
    testloader = DataLoader(testdataset, batch_size=32, shuffle=True)

    # the shape is 32, 28, 28. So 32 images each 28x28
    return trainloader, testloader
