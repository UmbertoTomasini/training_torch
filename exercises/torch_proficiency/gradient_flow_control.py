import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from dataclasses import dataclass
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, Dataset

@dataclass
class ModelArgs:
    in_channels: int = 3
    out_channels_1: int = 16
    out_channels_2: int = 32
    final_dim: int = 6
    num_classes: int = 10
    kernel_size: int=3
    stride: int =1


class SimpleCNN(nn.Module):

    def __init__(self, args = ModelArgs):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=args.in_channels,
            out_channels= args.out_channels_1,
            kernel_size=args.kernel_size,
            stride=args.stride
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=args.kernel_size,
            stride=args.stride
        )

        self.conv2= nn.Conv2d(in_channels=args.in_channels,
            out_channels= args.out_channels_2,
            kernel_size=args.kernel_size,
            stride=args.stride
        )
        self.pool2= nn.MaxPool2d(
            kernel_size=args.kernel_size,
            stride=args.stride
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((args.final_dim,args.final_dim))

        self.fc = nn.Linear(args.out_channels_2*args.final_dim*args.final_dim,args.num_classes)

        self.args = args

    def forward(self,x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.adaptive_pool(x)

        x = x.view(x.size(0),-1) #flatten
        x = self.fc(x)

        return x

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_avalaible() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])

    #load
    train_dataset = datasets.MNIST(root= './data', train =True, download=True,transform = transform)
    test_dataset = datasets.MNIST(root='./data', train = False, download =True, transform = transform)
    #dataloader
    batch_size = 1

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

    model = SimpleCNN(ModelArgs())

    loss_fn = nn.CrossEntropyLoss()

    learning_rate = 1e-4
    num_epochs = 5
   # Suppose you want an effective batch size of 4, but can only fit 1 in memory
    effective_batch_size = 4
    accumulation_steps = effective_batch_size // batch_size
    

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2)

    scaler = torch.cuda.amp.GradScaler(device)

    #for param in model.conv2.parameters():
    #    param.requires_grad = False

    def print_grad(modeule,grad_in, grad_out):
        print(grad_in)
        print(grad_out)

    model.conv1.register_full_backward_hook(print_grad)

    def scale_adapt_pool(module,input,output):
        return output*2

    model.adaptive_pool.register_forward_hook(scale_adapt_pool)


    for idx_batch, (x,y) in enumerate(test_dataloader):

        with torch.cuda.amp.autocast(device):
            y_pred = model(x)
            train_loss = loss_fn(y_pred,y)

        scaler.scale(train_loss).backward()

        #gradient accumulation
        if (idx_batch+1) % accumulation_steps :

            #gradient clipping
            torch.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

            #optimizer.step()
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item()**2
            total_norm = total_norm** 0.5

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step()



