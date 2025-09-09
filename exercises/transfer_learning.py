from importlib.metadata import requires
from pickletools import optimize
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import pdb
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os

import torchvision
from train_loop import train_loop

cudnn.benchmark  = True

data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    ),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    )
}

data_dir = 'data/'

image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) 
    for x in ['train', 'val']}

image_dataloaders = {x : DataLoader(image_datasets[x], batch_size = 4, shuffle= True, num_workers = 4)
    for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x])
    for x in ['train', 'val']}

class_names = image_datasets['train'].classes

num_classes = len(class_names)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f'device: {device}')


model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs,num_classes) # complete reset

model_ft.to(device)

criterion_ft = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.AdamW(model_ft.parameters(), lr = 1e-4, momentum = 0.9, weight_decay=1e-5)
scheduler_ft = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size =2)

train_loop(image_dataloaders['train'], model_ft, criterion_ft , optimizer_ft,
         scheduler_ft, batch_size =4)

##################################################
#LoRA
class LoRALayer(nn.Module):
    def __init__(self,in_features,out_features,rank=16):
        super().__init__()
        self.rank = rank
        self.scaling = 32 / rank

        self.lora_A = nn.Linear(in_features,rank,bias=False)
        self.lora_B = nn.Linear(rank,out_features,bias=False)

        nn.init.kaiming_uniform(self.lora_A.weight)
        nn.init.zeros(self.lora_B.weight)
    def forward(self,x):
        return self.lora_B(self.lora_A(x)) * self.scaling

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank = 16) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(original_layer.in_features, original_layer.out_features, rank)
        for param in self.original_layer.parameter():
            param.requires_grad = False
        
    def forward(self, x):
        return self.original_layer(x) + self.lora(x)

def apply_lora_to_model(model, rank = 16):
    for name, module in model.named_modules():
        if name == 'fc':
            parent = model
            setattr(parent,'fc',LoRALayer(module,rank))
            break
    return model



model = apply_lora_to_model(model,rank=16)



##################################################
model_conv = torchvision.models.resnet18(weights= 'IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs,2)

model_conv = model_conv.to(device)

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr =1e-3)






# freeze previous layers
