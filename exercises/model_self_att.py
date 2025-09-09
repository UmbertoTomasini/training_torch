from turtle import forward
from numpy import shape
from sympy.geometry.polygon import x
import torch
import torch.nn as nn
import torch.nn.functional as F 
from dataclasses import dataclass
from typing import Optional
import math

from llama2.model import apply_rotary_embeddings

@dataclass
class ModelArgs:
    dim: int = 4096
    
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: Optional[int] = None

    vocab_size: int = -1

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

    ff_dim: int = 8192

    eps_norm = 1e-5

def repeat_kv(kv: torch.Tensor, n_rep: int):
    if n_rep ==1:
        return kv
    else:
        batch_size, seq_len, n_heads, head_dim = x.shape
        return (
            kv[:,:,:,None,:]
            .expand(batch_size,seq_len, n_heads,n_rep,head_dim)
            .reshape(batch_size,seq_len, n_heads*n_rep,head_dim)
        )

class RMSNorm(nn.Module):
    def __init__(self, dim, args = ModelArgs, eps: float = 1e-5) -> None:
        super().__init__()
        self. weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim
    def forward(self, x: torch.Tensor):
        #[batch_size, seq_len, dim]
        x *= torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
        x = x * self.weight
        return x



class SelfAttention(nn.Module):

    def __init__(self, args = ModelArgs) -> None:
        super().__init__()

        self.n_heads_q = args.n_heads
        self.n_heads_kv = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        self.n_rep = self.n_heads_q // self.n_heads_kv

        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads_q*self.head_dim,bias=False)
        self.wk = nn.Linear(args.dim, self.n_heads_kv*self.head_dim,bias=False)
        self.wv = nn.Linear(args.dim, self.n_heads_kv*self.head_dim,bias=False)

        self.wo = nn.Linear(self.head_dim * args.n_heads, args.dim, bias = False)

        self.cache_k = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_heads_kv,self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_heads_kv,self.head_dim))

    def forward(self,x: torch.Tensor, start_pos: int, use_cache: bool = False):

        batch_size, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        #multi-head
        xq = xq.view(batch_size,seq_len,self.n_heads_q,self.head_dim)

        xk = xq.view(batch_size,seq_len,self.n_heads_kv,self.head_dim)
        xv = xq.view(batch_size,seq_len,self.n_heads_kv,self.head_dim)

        # apply rotary embeddings in case
        # xq = apply_rotary_embeddings(xq,freqs_complex, device = x.device)
        # xk = apply_rotary_embeddings(xq,freqs_complex, device = x.device)

        if use_cache:
            self.cache_k[:batch_size, start_pos : start_pos + seq_len ] = xk
            self.cache_v[:batch_size, start_pos : start_pos + seq_len ] = xv
            keys = self.cache_k[:batch_size, : start_pos + seq_len]
            values = self.cache_v[:batch_size, start_pos +seq_len]
        else:
            #training mode
            keys = xk
            values = xv
        
        #expand
        keys = repeat_kv(keys,self.n_rep)
        values = repeat_kv(values,self.n_rep)

        #att matrix
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        att_matrix = torch.matmul(xq,keys.transpose(-2,-1)) / math.sqrt(self.head_dim)

        if not use_cache:
            causal_mask = torch.tril(torch.ones_like(att_matrix), device = x.device)
            att_matrix = att_matrix.masked_fill(causal_mask==0, float('-inf'))

        scores = F.softmax(att_matrix, dim =-1).type_as(xq) #[B, num_heads, T, T]

        #[B, num_heads, T, head_dim] --> [B, T, num_heads, head_dim] --> [B, T, num_heads*head_dim]
        output = torch.matmul(scores, values).transpose(1,2).contiguous().view(batch_size,seq_len,-1)

        output = self.wo(output)

        return output

class FeedForward(nn.Module):

    def __init__(self,args: ModelArgs):
        super().__init__()


        self.up_projection = nn.Linear(args.dim, args.ff_dim,bias=False)
        self.relu = nn.ReLU()
        self.down_projection = nn.Linear(args.ff_dim,args.dim, bias=False)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,h: torch.Tensor):
        h = self.up_projection(h)
        h = self.relu(h)
        h = self.down_projection(h)
        h = self.dropout(h)
        return h

class EncoderBlock(nn.Module):
    def __init__(self, args=ModelArgs) -> None:
        super().__init__()

        self.norm_before_attn = RMSNorm(args.model_dim,eps =args.eps_norm)
        self.attn = SelfAttention(ModelArgs)

        self.norm_before_ff = RMSNorm(args.model_dim,eps =args.eps_norm)
        self.ff = FeedForward(ModelArgs)
    
    def forward(self, h: torch.Tensor, start_pos: int,):
        h  = h + self.attn(self.norm_before_attn(h), start_pos)
        h = h + self.ff(self.norm_before_ff(h))
        return h

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert vocab_size!=-1, "vocab size must be set"

        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size,args.dim)
        self.pos_embedding = nn.Embedding(args.max_seq_len,args.dim)

        self.n_layers = args.n_layers

        self.layers = nn.ModuleList(
            [EncoderBlock(args) for _ in range(self.n_layers)]
        )

        self.norm = RMSNorm(args.dim, eps = args.eps_norm)

        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=False)
    
    def forward(self, x: torchTensor, start_pos: int):
        #[batch size, seq_len]
        batch_size, seq_len = x.shape
        #[batch size, seq_len] --> [batch_size, seq_len, dim]
        x = self.tok_embeddings(x)

        position_ids = torch.arange(start_pos , start_pos+seq_len, device = x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
         #[batch size, seq_len] --> [batch_size, seq_len, dim]
        x += self.pos_embedding(position_ids)

        for layer in self.layers:
            x = layer(x,start_pos)

        x = self.norm(x)
        x = output(x).float()

        return x


    






