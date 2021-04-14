import torch
import numpy as np
from tqdm import tqdm

# Computes Hessian vector product
def Hvp(loss, v, model, device, data_loader):
    Hv = torch.zeros_like(v)
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        L = loss(output, target) * data.size(0) / len(data_loader.dataset)
        grad = torch.autograd.grad(L, model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.reshape(-1) for g in grad if g is not None])
        prod = torch.dot(grad_vec, v)
        grad = torch.autograd.grad(prod, model.parameters())
        Hv += torch.cat([g.reshape(-1) for g in grad if g is not None])
    if device.type == "xla":
        import torch_xla.core.xla_model as xm
        Hv = xm.mesh_reduce("Hv", Hv, np.sum)
    return Hv

# Computes top eigensubspace of Hessian via power series
def subspace(loss, model, device, data_loader, dim, iters):
    model.train()
    m = sum(p.numel() for p in model.parameters())
    Q = torch.randn((m, dim)).to(device)
    for i in tqdm(range(iters)):
        HV = torch.zeros((m, dim))
        for j in tqdm(range(dim), leave=False):
            HV[:,j] = Hvp(loss, Q[:,j], model, device, data_loader)
        Q, R = torch.qr(HV)
    return Q.data.numpy(), torch.diag(R).data.numpy()

# Computes complete Hessian matrix
def hessian(loss, model, device, data_loader):
    model.train()
    m = sum(p.numel() for p in model.parameters())
    H = torch.zeros(m, m)
    for i in tqdm(range(m)):
        v = torch.zeros(m)
        v[i] = 1.0
        H[i] = Hvp(loss, v, model, device, data_loader)
    return H.data.numpy()
