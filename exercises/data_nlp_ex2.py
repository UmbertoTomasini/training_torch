import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import List, Dict, Optional
import torch.nn.utils.rnn as rnn_utils


def encode_and_pad(texts: List, pad_value: int = 0):
    all_words = []
    for text in texts:
        text = text.split()
        all_words+=text
    all_words = list(set(all_words))

    all_words.sort()
    dic_encode= {}
    for idx_word, word in enumerate(all_words):
        dic_encode[word] = idx_word +1


    encoded_texts = []
    att_masks = []
    for text in texts:
        encoded_text = []
        for word in text.split():
            encoded_text.append(dic_encode[word])
        att_mask = [1]*len(encoded_text)

        encoded_text = torch.tensor(encoded_text, dtype=torch.long)
        att_mask = torch.tensor(att_mask, dtype=torch.bool)

        encoded_texts.append(encoded_text)
        att_masks.append(att_mask)
    
    padded_encoded_texts = rnn_utils.pad_sequence(encoded_texts, batch_first = True, padding_value = pad_value)
    padded_att_masks = rnn_utils.pad_sequence(att_masks, batch_first = True, padding_value = False)

    return padded_encoded_texts, padded_att_masks

class SimpleDataset(Dataset):
    def __init__(self, texts: List, labels: List) -> None:
        super().__init__()

        assert len(texts)==len(labels), "Labels and Texts mismatch in number"

        padded_encoded_texts, padded_att_masks = encode_and_pad(texts, pad_value = 0)

        self.texts = padded_encoded_texts
        self.att_masks = padded_att_masks
        self.labels = labels


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return (self.texts[index], self.att_masks[index], self.labels[index])



