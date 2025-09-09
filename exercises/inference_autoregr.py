import torch
from torch._library.infer_schema import SUPPORTED_RETURN_TYPES
import torch.nn as nn
import torch.nn.functional as F 
import math
from model_self_att import Transformer, ModelArgs
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import json
from typing import List, Optional

class InferenceModel():

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
    
    @staticmethod
    def build(checkpoint_dir: str, tokenizer_path: str, max_batch_size: int, max_seq_len, device: str):

        checkpoint = model.load(checkpoint_dir, map_location = "auto")

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)

        with open("params.json") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_batch_size = max_batch_size
            max_seq_len = max_seq_len,
            vocab_size = tokenizer.vocab_size(),
            **params
        )

        model = Transformer(model_args).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        return InferenceModel(model,tokenizer,model_args)

    def text_generation(self, prompts: List[str], temperature: float = 0.6, top_p: int = 0.9, k_beams:Optional[int]=None,top_k: Optional[int]=None,max_gen_len: Optional[int]=None):

        batch_size = len(prompts)
        assert batch_size <= self.args.max_batch_size, "too many prompts"

        if max_gen_len is None:
            max_gen_len = self.args.max_gen_len -1

        prompt_tokens = [self.tokenizer.encode(prompt, out_type = int, add_bos = True, add_eos=False) for prompt in prompts]

        max_prompt_len = max(len(prompt) for prompt in prompts)
        assert max_prompt_len <= self.args.max_seq_len

        total_len = min(self.args.max_seq_len, max_prompt_len+max_gen_len)

        pad_id = self.tokenizer.pad_id()

        tokens = torch.full((batch_size,total_len), pad_id, dtype = torch.long, device = device)

        for idx_prompt,prompt in enumerate(prompt_tokens):
            tokens[idx_prompt, : len(prompt)] = torch.tensor(prompt,dtype = torch.long,device = device)

        mask_prompt_tokens = tokens != pad_id

        eos_reached = torch.tensor([False]*batch_size,device=device)

        for cur_pos in tqdm(range(1,total_len),desc="Generating tokens"):
            with torch.no_grad():
                logits = self.model(tokens[:,cur_pos-1:cur_pos],cur_pos)

            if temperature >0:
                probs = F.softmax(logits[:,-1,:]/temperature, dim =-1)

                next_token = self.sample_top_p_top_k(probs,top_p,top_k)
            else:
                next_token = torch.argmax(logits,dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(mask_prompt_tokens[:,cur_pos], tokens[:,cur_pos], next_token)

            tokens[:,cur_pos] = next_token

            eos_reached |= (next_token==self.tokenizer.eos_id()) & (mask_prompt_tokens[:,cur_pos])
            if all(eos_reached):
                break
        
        out_tokens = []
        out_texts = []
        for idx_prompt, prompt_token in range(batch_size):
            if self.tokenizer.eos_id() in prompt_token:
                eos_idx = prompt_token.index(self.tokenizer.eos_id())
                prompt_token = prompt_token[:eos_idx]
            out_tokens.append(prompt_token)
            out_texts.append(self.tokenizer.decode(prompt_token))
        
        return out_tokens, out_texts

    def sample_top_p_top_k(probs,top_p = 0.9,top_k: Optional[int]=None):
        #[batch_size, vocab_size]
        sorted_probs, idx_sorted_probs = torch.sort(probs,dim=-1,descending=True)

        if top_k is not None:
            sorted_probs[:,top_k:] = 0.0

        cum_probs = torch.cumsum(support_probs,dim =-1)

        mask = (cum_probs-sorted_probs)> top_p
        sorted_probs[mask] = 0
        sorted_probs.div(sorted_probs.sum(dim=-1,keepdim = True))

        next_tokens = torch.multinomial(sorted_probs, num_samples = 1)

        next_tokens = torch.gather(idx_sorted_probs, -1, next_token,)

        return next_tokens






if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inf_model = InferenceModel.build(
        checkpoint_dir, 
        tokenizer_path, 
        max_batch_size = 16, 
        max_seq_len = 128, 
        device = device)

    prompts = [""]

    out_tokens, out_text = model.text_generation(prompts, max_gen_len = 64)
    assert len(out_tokens)==len(out_text)
    for i in range(len(out_text)):
        print(f"{out_text[i]}")
        print("-"*50)




