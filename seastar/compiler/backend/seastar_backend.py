import torch
import ctypes
from abc import ABC, abstractmethod

from seastar.compiler.backend.kernel_wrapper import KernelWrapperTorch

# class KernelWrapper(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, executor, kid, kernel_args, rets, *args):
#         ctx.backward_cache = executor, kid
#         ret = executor.forward_cb(kid, kernel_args, rets, args)
#         return ret

#     @staticmethod
#     def backward(ctx, *gradout):
#         executor, kid = ctx.backward_cache
#         return (None, None, None, None) + executor.backward_cb(kid, gradout)


# def backend_cb(executor):
#     def new_zeros_call_back(size, dtype, device, requires_grad=True):
#         return torch.zeros(
#             size=size, dtype=dtype, device=device, requires_grad=requires_grad
#         )

#     def tensor_raw_ptr(tensor):
#         import ctypes

#         return ctypes.c_void_p(tensor.data_ptr())

#     executor.set_new_zeros_cb(new_zeros_call_back)
#     executor.set_raw_ptr_cb(tensor_raw_ptr)
#     return executor.execute(KernelWrapper)

#################################################################################################

class SeastarBackend(ABC):
    def __init__(self):
        self.backend_name = None
        self.kernel_wrapper = None
        
    @abstractmethod
    def new_zeros_call_back(self, size, dtype, device, requires_grad=True):
        pass
    
    @abstractmethod
    def tensor_raw_ptr(self, tensor):
        pass
    
    def backend_cb(self, executor):
        executor.set_new_zeros_cb(self.new_zeros_call_back)
        executor.set_raw_ptr_cb(self.tensor_raw_ptr)
        
        return executor.execute(self.kernel_wrapper)
    
class SeastarBackendTorch(SeastarBackend):
    def __init__(self):
        super().__init__()
        self.backend_name = "torch"
        self.kernel_wrapper = KernelWrapperTorch
        
    def new_zeros_call_back(self, size, dtype, device, requires_grad=True):
        return torch.zeros(
            size=size, dtype=dtype, device=device, requires_grad=requires_grad
        )
        
    def tensor_raw_ptr(self, tensor):
        return ctypes.c_void_p(tensor.data_ptr())