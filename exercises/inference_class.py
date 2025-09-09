import torch
import math
import torch.nn as nn
import torch.nn.functional as F 
from typing import Optional
from model_mlp_classifier import MLPClassifier, ModelArgs
from data_nlp_ex2 import SimpleDataset
from torch.utils.data import DataLoader




if __name__ =="__main__":

    model = MLPClassifier(ModelArgs())
    model.load_state_dict(torch.load('model_dummy.pth',weights_only=True))

    model.eval()

    texts_val = ['Pizza Pasta']
    label_val = [0]
    val_dataset = SimpleDataset(texts_val, label_val)

    batch_size_val =1

    val_dataloader = DataLoader(val_dataset, batch_size = batch_size_val, shuffle = True)

    num_tot_val = len(val_dataset)

    class_loss = nn.CrossEntropyLoss()

    acc = 0
    class_loss_acc = 0

    with torch.no_grad():
        for batch, (x, _, y) in enumerate(val_dataloader):
            logits_pred = model(x)

            #accuracy
            pred = torch.argmax(logits_pred,dim=-1)
            acc+= (pred==y).sum().item()

            #class loss
            class_loss_acc += class_loss(logits_pred, y) * x.size(0)

    acc /= num_tot_val
    class_loss_acc /=num_tot_val

    print(f"Acc: {acc}")
    print(f"Loss: {class_loss_acc}")       




