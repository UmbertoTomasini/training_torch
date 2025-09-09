import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

learning_rate = 1e-3
batch_size = 64
epochs = 5

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

def train_loop(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)

    model.train()

    for batch , (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred,y)

        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 ==0:
            loss, current = loss.item(), batch*batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader,model,loss_fn):
    model.eval()
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0

    with torch.no_grad():
        for x,y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred,y)
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        
    test_loss /= num_batches
    correct /=size
    print(f"Testerror: \n Accuracy: {100*correct}\n Avg loss: {test_loss}\n")

for t in range(epochs):
    print("Epoch: {t}")
    train_loop(train_dataloader,model,loss_fn,optimizer)
    test_loop(test_dataloader,model,loss_fn)

torch.save(model.state_dict(),'model_fc_2nn.pth')

model_2 = NeuralNetwork()
model.load_state_dict(torch.load('model_fc_2nn.pth',weights_only=True))

model.eval()
