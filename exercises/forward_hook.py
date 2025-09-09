import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10,20),
    nn.ReLU(),
    nn.Linear(20,1)
)

def relu_hook_fn(module, input, output):
    mod_output = output.clamp(max=0.5)
    return mod_output

hook = model[1].register_forward_hook(relu_hook_fn)


