import torch
import torch.nn.functional as f
import torch.nn as nn
from model_mlp_classifier import ModelArgs,MLPClassifier
from data_nlp_ex2 import SimpleDataset
from torch.utils.data import DataLoader


def grad_hook(module, grad_input, grad_output):
    if module.weight.grad is not None:
        norm = module.weight.grad.norm().item()
        print(f"{norm} target")

def get_grad_norm(model: torch.nn.Module):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
    total_norm = total_norm **(1. / 2)
    return total_norm

def print_grad_parameters(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}, grad norm: {param.grad.data.norm().item()}")

def train_loop(train_dataloader, model, loss_fn, optimizer, scheduler, batch_size):
    model.train()
    size = len(train_dataloader)

    

    for batch, (data, mask, labels) in enumerate(train_dataloader):

        labels_pred = model(data)

        train_loss = loss_fn(labels_pred,labels)

        train_loss.backward()

        grad_norm = get_grad_norm(model)
        print_grad_parameters(model)

        
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm = 1.0)

        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()

        if batch%1==0:
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"Loss: {train_loss.item()}, lr = {current_lr}, norm_gradients = {total_norm}, [{batch*batch_size}/{size}]")






if __name__ =="__main__":
    torch.manual_seed(0)

    #training data!

    with open("data/list_strings.txt") as f:
        texts = f.read()
    texts = texts.split("\n")
    #embedder = SentenceTransformer('assl-MiniLM-L6-v2')

    #embeddings = embedder.encode(text)

    labels = [1,0]




    training_data = SimpleDataset(texts,labels)

    train_dataloader = DataLoader(training_data,batch_size=1)

    #model
    model = MLPClassifier(args = ModelArgs())

    #training stuff
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    epochs = 5
    batch_size = 1

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2)

    target_module = model.layer[0].attention.wq
    handle = target_module.register_full_backward_hook(grad_hook)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n--------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, scheduler, batch_size)
    print("Done!")

