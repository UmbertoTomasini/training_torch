import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from model_self_att import Transformer, ModelArgs
from sentencepiece import SentencePieceProcessor

if __name__ == "__main__":


    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    args = ModelArgs()
    args.vocab_size = tokenizer.vocab_size()

    model = Transformer(args)
    checkpoint = torch.load("checkpoint.pth")
    model.load_state_dict(checkpoint)

    #print all modules
    for name, module in model.named_modules():
        print(name,module)
    for name, module in model.named_children():
        print(name,module)


    #I want to get all attention maps
    
    attention_maps = []

    def hook_fn(module, input, output):
        attention_maps.append(output.detach().cpu())

    handles = []
    for idx_layer in range(args.n_layers):
        handle = model.layers[idx_layer].attention.register_forward_hook(hook_fn)
        handles.append(handle)

    output = model(input_tensor)

    for handle in handles:
        handle.remove()

    
