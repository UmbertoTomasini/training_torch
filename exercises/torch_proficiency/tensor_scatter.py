import torch
import math
import pdb


M =20 #size target tensor
N = 5 #number of updates/operations to perform among the M dimensions
D = 10 
#sparse: just updating N positions out of M, scattered

updates = torch.randn([N,D])
indices = torch.randint(low = 0, high = M, size = [N]) #indices on which the operations are done

target = torch.randn([M,D])



# works only IF there are no duplicate indices in indices
# target[indices,:] += updates
# if there are duplicates:
for i in range(N):
    target[indices[i]] += updates[i]

# otherwise use custom function
target.scatter_add_(0, indices.unsqueeze(-1).expand(-1,D), updates)

