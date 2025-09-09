import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F

ds = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: F.one_hot(torch.tensor(y), num_classes=10).float())
)

