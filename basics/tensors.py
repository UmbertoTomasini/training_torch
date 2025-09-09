import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n{x_ones}\n")

x_rand = torch.rand_like(x_data,dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)

print(f"Shape: {rand_tensor.shape}")
print(f"Shape: {rand_tensor.dtype}")
print(f"Shape: {rand_tensor.device}")

if torch.accelerator.is_available():
    rand_tensor = rand_tensor.to(torch.accelerator.current_accelerator())

torch_cat = torch.cat([rand_tensor,rand_tensor],dim=-1)
print(torch_cat.shape)

t1 = rand_tensor@ rand_tensor.T
t1_copy = rand_tensor.matmul(rand_tensor.T)

z1 = rand_tensor*rand_tensor
z1_copy_1 = rand_tensor.mul(rand_tensor)


agg = rand_tensor.sum()
agg_item = agg.item()




