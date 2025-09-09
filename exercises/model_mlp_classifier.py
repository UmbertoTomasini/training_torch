import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelArgs:
    input_dim: int = 4096
    n_layers: int = 2
    hidden_dim: int = 8192
    max_batch_size: int = 32
    num_classes: int = 2
    vocab_size: int = 100


class MLPClassifier(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args

        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim = args.input_dim)

        self.first_layer  = nn.Sequential(
            nn.Linear(args.input_dim, args.hidden_dim),
            nn.ReLU()
            )

        if args.n_layers >2:
            self.layers = nn.Sequential(*[
                nn.Linear(args.hidden_dim,args.hidden_dim),
                nn.ReLU()
            ]*(args.n_layers-2))
        
        self.last_layer  = nn.Linear(args.hidden_dim, args.num_classes)

        

    def forward(self, x: torch.Tensor):
        # Create new tensor with correct type
        x = torch.tensor(x, dtype=torch.long)
        
        # Now embed the tokens
        x = self.embedding(x).mean(1)
        
        batch_size, input_dim = x.size()
        
        assert batch_size <= self.args.max_batch_size, "Over max batch size"
        assert input_dim == self.args.input_dim, "Not correct input dim, mismatch"
        
        out = self.first_layer(x)
        if self.args.n_layers > 2:
            out = self.layers(out)
        out = self.last_layer(out)
        
        return out

if __name__ == "__main__":

    args = ModelArgs
    model = MLPClassifier(args)

    x = torch.randn(16,args.input_dim)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")



