import torch
import torch.nn as nn
import pdb


x = torch.tensor([2,4,1])

vocab_size = 10
dim = 32

embed = nn.Parameter(torch.randn(vocab_size,dim))

embedded_x = embed[x]

embedded_x_2 = torch.gather(embed, 0, x.unsqueeze(1).expand(-1,dim))

print(embedded_x)
print(embedded_x_2)


print(torch.equal(embedded_x, embedded_x_2))
print(torch.allclose(embedded_x, embedded_x_2, rtol=1e-5,atol=1e-8))
print(torch.isclose(embedded_x, embedded_x_2, rtol=1e-5,atol=1e-8))




