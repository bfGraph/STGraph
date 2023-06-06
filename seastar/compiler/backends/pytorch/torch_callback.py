import torch
import ctypes

from seastar.compiler.backends.callback import SeastarBackend
from seastar.compiler.backends.pytorch.torch_kernel_wrapper import KernelWrapperTorch 

class SeastarBackendTorch(SeastarBackend):
    ''' Seastar backend using PyTorch framework'''
    def __init__(self):
        super().__init__()
        self.backend_name = "torch"
        self.backend_module = torch
        self.kernel_wrapper = KernelWrapperTorch
        
    def new_zeros_call_back(self, size, dtype, device, requires_grad=True):
        return torch.zeros(
            size=size, dtype=dtype, device=device, requires_grad=requires_grad
        )
        
    def tensor_raw_ptr(self, tensor):
        return ctypes.c_void_p(tensor.data_ptr())