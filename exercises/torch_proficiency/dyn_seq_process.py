import torch
import math
import pdb

max_size = 10
num_tensors = 5
sizes = torch.randint(0,max_size,[num_tensors])

list_tensors = []
for size_tensor in sizes:
    
    list_tensors.append(torch.randn(size=[size_tensor.item()]))

def dummy_fun(seq: torch.Tensor):
    seq += 2
    return seq
    
processed_tensors = []
for single_tensor in list_tensors:
    processed_tensors.append(dummy_fun(single_tensor))




