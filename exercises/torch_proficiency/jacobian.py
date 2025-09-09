import torch
from torch.autograd.functional import jacobian


def compute_jacobian(func,x):

    x = x.requires_grad_()
    y = func(x)

    jacobian = torch.zeros([y.shape[0], x.shape[0]], dtype = x.dtype, device = x.device)

    for i in range(y.shape[0]):

        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1

        grads = torch.autograd.grad(y,x,grad_outputs=grad_outputs,retain_graph=True)[0]

        jacobian[i] = grads

    return jacobian

x = torch.tensor([1.0,2.0],requires_grad=True)
def f(x):
    return torch.stack([x[0]+x[1],x[0]*x[1]])

J = compute_jacobian(f,x)

print(J)
