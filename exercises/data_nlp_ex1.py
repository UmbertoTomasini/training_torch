import torch 
import pdb
import string
from typing import List, Dict

from torch.nn.functional import pad


vocab = string.ascii_letters + string.digits + string.punctuation + " "
char_to_idx = {c: ord(c) for c in vocab}
pad_token = '<PAD>'
char_to_idx[pad_token] = max(char_to_idx.values())*2



def encode_and_pad_strings_from_scratch(strings: List, char_to_idx: Dict):
    max_len_strings = max(len(s) for s in strings)

    text_indexed = []
    attention_masks = []
    for string in strings:
        string_indexed = [char_to_idx[c] for c in string]
        string_indexed += [char_to_idx[pad_token]]*(max_len_strings-len(string_indexed))
        text_indexed.append(torch.tensor(string_indexed,dtype = torch.long))
        att_mask = [1] *len(string) + [0]* (max_len_strings - len(string))
        attention_masks.append(torch.tensor(att_mask,dtype=torch.bool))
    padded_tensor = torch.stack(text_indexed)
    attention_masks = torch.stack(attention_masks)
    return padded_tensor, attention_masks


with open("data/list_strings.txt") as f:
    text = f.read()
text = text.split("\n")

encoded_text, attention_masks = encode_and_pad_strings_from_scratch(text, char_to_idx)

pdb.set_trace()
    

