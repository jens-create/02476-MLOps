from data import mnist
from torchvision.utils import save_image

trainloader, testloader = mnist()  # data
batch_size = 32
x_dim = 784

for batch_idx, (x, _) in enumerate(testloader):
    x = x.view(batch_size, x_dim)
    break

save_image(x.view(batch_size, 1, 28, 28), "orig_data.png")
