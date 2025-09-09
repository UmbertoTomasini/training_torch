import os
from turtle import forward

from numpy.linalg import LinAlgError
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
print(f"Using {device} device")

class FC_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack= nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
    def forward(self,x):
        x= self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = FC_net().to(device)

X= torch.rand(1,28,28,device=device)
logits = model(X)

pred_prob = nn.Softmax(dim=1)(logits)

y_pred = pred_prob.argmax(1)
print(f"Pred prob: {y_pred}")


