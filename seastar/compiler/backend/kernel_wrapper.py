"""
from abc import ABC, abstractmethod

import torch
from rich import inspect

class KernelWrapper():
    def forward(ctx, executor, kid, kernel_args, rets, *args):
        ctx.backward_cache = executor, kid
        ret = executor.forward_cb(kid, kernel_args, rets, args)
        return ret

    def backward(ctx, *gradout):
        executor, kid = ctx.backward_cache
        return (None, None, None, None) +  executor.backward_cb(kid, gradout)

class TorchKernelWrapper(torch.autograd.Function):
    pass

t = TorchKernelWrapper
inspect(t, all=True)
"""