import torch
import ctypes

from stgraph.compiler.backend.callback import STGraphBackend
from stgraph.compiler.backend.pytorch.torch_kernel_wrapper import KernelWrapperTorch 

class STGraphBackendTorch(STGraphBackend):
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