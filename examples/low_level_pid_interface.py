from __future__ import annotations

import torch

import helion.language as hl


def add(x, y):
    block_size = 1024
    out = torch.empty_like(x)
    for pid in hl.grid(hl.cdiv(x.shape[0], block_size)):
        block = hl.masked_block(pid, block_size, x.shape[0])
        out[block] = x[block] + y[block]
    return out
