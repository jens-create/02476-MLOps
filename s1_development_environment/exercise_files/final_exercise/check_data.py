from pytorch_lightning import Trainer  # somehow necessary?
from data import mnist
from torchvision.utils import save_image


# model = MyAwesomeModel()  # this is our LightningModule
trainloader, testloader = mnist()  # data
batch_size = 32
x_dim = 784

for batch_idx, (x, _) in enumerate(testloader):
    x = x.view(batch_size, x_dim)
    # x = x.to(DEVICE)
    # x_hat, _, _ = model(x)
    save_image(x.view(batch_size, 1, 28, 28), "orig_data" + str(batch_idx) + ".png")
