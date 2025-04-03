import helion
import helion.language as hl
import torch


@helion.kernel
def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel
def torch_ops_pointwise(x, y):
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sigmoid(torch.add(torch.sin(x[tile]), torch.cos(y[tile])))
    return out
