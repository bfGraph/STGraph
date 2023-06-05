import torch
import ctypes
from abc import ABC, abstractmethod

from seastar.compiler.backend.kernel_wrapper import KernelWrapperTorch

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
    ''' Seastar backend using PyTorch framework'''
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
    
class SeastarBackendTF(SeastarBackend):
    ''' Seastar backend using Tensorflow framework'''
    pass

class SeastarBackendMXNet(SeastarBackend):
    ''' Seastar backend using MXNet framework'''
    pass