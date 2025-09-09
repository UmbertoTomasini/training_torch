from audioop import bias
from pickletools import read_decimalnl_long
from tracemalloc import start
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int=4096
    n_layers: int = 32
    n_heads: int = 32 #number heads for queries
    n_kv_heads: Optional[int] = None #number heads for the keys and values

    vocab_size: int = 1 #set with tokenizer

    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    norm_eps: float = 1e-5

    #needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_frequencies(head_dim:int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim %2 == 0, "Dimension must be divisible by 2" 

    #Build theta params
    # According to formula theta_i = 10000^(-2(i-1)/dim) for i = [1,2,...,dim/2]
    # Shape: (Head_dim / 2)

    theta_numerator = torch.arange(0,head_dim//2).float()
    # Shape: Head_dim / 2
    theta_denominator = 1.0/(theta**(theta_numerator / head_dim)).to(device)

    # Construct the positions (m parameter)
    # Shape (Seq_Len)
    m = torch.arange(0,seq_len,device = device)

    #Multiply each theta by each position using outer product
    # Shape: (seq_len) outer product *(Head_Dim/2) -> (Seq_Len, Head_Dim /2)
    freqs = torch.outer(m,theta_denominator).float()
    # We can compute a complex numbers in the polar form c = R*exp(i*m*theta), where R=1 as follows
    # Shape: (Seq_Len, Head_Dim /2) --> (Seq_Len, Head_Dim .2)
    freqs_complex = torch.polar(torch.ones_like(freqs),freqs)

    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, seq_len, H, Head_dim), H is the number of heads
    # -> 
    # (B, seq_len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
    # (Seq_len, Head_dim/2) 
    # -> 
    # (1, Seq_len, 1, Head_Dim,2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # (B, seq_len, H, Head_Dim / 2) * (1, Seq_len, 1, head_dim/2) = (B, seq_len, H, Head_dim / 2)
    x_rotatated = x_complex * freqs_complex

    #(B, seq_len, H, head_dim/2) -> (B, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotatated)

    #flatten it
    #(B, seq_len, H, head_dim/2, 2) -> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep ==1:
        return x
    else:
        return (
            #(B,seq_len, n_kv_heads,1,head_dim)
            x[:,:,:,None,:]
            .expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim)
            .reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim)
        )
        

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        #gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_len, Dim) * (B, seq_len,1) -> (B, seq_len, dim)
        # rsqrt(x): 1/sqrt(x)
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self,x:torch.Tensor):
        # (Dim) * (B, seq_len, dim) = (B, seq_len,dim)
        return self.weight * self._norm(x.float()).type_as(x )

class SelfAttention(nn.Module):
    def __init__(self,args: ModelArgs) -> None:
        super().__init__()
            # Number heads keys, values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
         # how many times the keys and values should be repated to match the head of the queries
        self.n_rep = self.n_heads_q //self.n_kv_heads
        # Dimension of each head
        self.head_dim = args.dim //args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads*self.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads*self.head_dim,args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim))


    def forward(self,x:torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape #(B, 1, Dim)
        
        # (B,1,Dim) -> (B,1, H_Q * Head_dim), no diff actually
        xq = self.wq(x)
        # (B,1,Dim) -> (B,1, H_KV * Head_dim), can be different
        xk = self.wk(x)
        xv = self.wv(x)

        #(B,1, H_Q * Head_dim) - > (B,1, H_Q , Head_dim)
        xq = xq.view(batch_size,seq_len,self.n_heads_q,self.head_dim)

        #(B,1, H_KV * Head_dim) - > (B,1, H_KV , Head_dim)
        xk = xk.view(batch_size,seq_len, self.n_kv_heads,self.head_dim)
        xv = xv.view(batch_size,seq_len, self.n_kv_heads,self.head_dim)

        # no change shape tensors
        xq = apply_rotary_embeddings(xq, freqs_complex,device =xq.device) # same shape
        xk = apply_rotary_embeddings(xk, freqs_complex,device =xk.device) # same shape

        # Replace the entry in the cache for this token
        self.cache_k[:batch_size,start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size,start_pos:start_pos+seq_len] = xv

        # Retrieve all the cacjed keys and values so far
        # (B, seq_len_KV (up to start_pos+1), H_KV, head_dim)
        keys = self.cache_k[:batch_size,0:start_pos+seq_len]
        values = self.cache_v[:batch_size,0:start_pos+seq_len]

        # Repeat the heads of the K and V to reach the number of heads of the queries
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values,self.n_rep)

        #(B, 1, H_Q, Head_dim ) --> (B, H_Q, 1, Head_dim)
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # (B, H_Q, 1, Head_dim) @ (B, H_Q, Head_dim, 1) --> (B, H_Q, 1,1)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(),dim=-1).type_as(xq)

        # (B, H_Q, 1,1) @ (B, H_Q, 1, Head_dim) -->(B, H_1, 1, Head_Dim)
        out = torch.matmul(scores, values)

        #remove heads by concateting
        out = (out.transpose(1,2).contiguous().view(batch_size,seq_len,-1))
        return self.wo(out) #(B,1,Dim)


class FeedForward(nn.Module):

    def __init__(self,args: ModelArgs) -> None:
        super().__init__()

        hidden_dim = 4*args.dim
        hidden_dim = int(2*hidden_dim/3)
 
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        #round the hidden_dim to the nearest multiple of the multiple of parameter
        hidden_dim = args.multiple_of * ((hidden_dim+args.multiple_of-1)//args.multiple_of)  # Fixed: was 'hidden'
       
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)  # Fixed: was 'hidden'
        self.w2 = nn.Linear(hidden_dim,args.dim, bias =False)  # Fixed: was 'hidden_dim'
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)  # Fixed: was 'hidden_dim'

    def forward(self, x: torch.Tensor):  # Fixed: was missing colon
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x







        
class EncoderBlock(nn.Module):

    def __init__(self,args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        #normalization BEFORE the self attention
        self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)
        #normalizatoin AFTER the self attention
        self.ffn_norm = RMSNorm(args.dim, eps = args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, dim)+ (B, seq_len, dim ) - >(B, seq_len, dim)
        h = x+self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out 

class Transformer(nn.Module):

    def __init__(self,args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size!=-1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size

        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size,args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim,eps=args.norm_eps)

        self.output = nn.Linear(args.dim,self.vocab_size,bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads,self.args.max_seq_len*2,device =self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):

        # (B,seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        #(B,Seq_Len) --> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        #Retrieve the pairs (m,theta) correspoding to positions [start_pos, start_pos+ seq_len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h,start_pos,freqs_complex)
        
        h = self.norm(h)

        output = self.output(h).float()
        return output