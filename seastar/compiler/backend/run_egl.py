"""
from abc import ABC, abstractmethod
from kernel_wrapper import TorchKernelWrapper

import torch
import ctypes

class SeastarBackend(ABC):
    
    def __init__(self):
        self.kernel_wrapper = None
    
    @abstractmethod
    def new_zeros_call_back(size, dtype, device, requires_grad=True):
        pass
    
    @abstractmethod
    def tensor_raw_ptr(tensor):
        pass
    
    def run_egl(self, executor):
        executor.set_new_zeros_cb(self.new_zeros_call_back)
        executor.set_raw_ptr_cb(self.tensor_raw_ptr)
        
        return executor.execute(self.kernel_wrapper)
    
class TorchSeastarBackend(SeastarBackend):
    def __init__(self):
        super().__init__()
        self.kernel_wrapper = TorchKernelWrapper
        
    def new_zeros_call_back(size, dtype, device, requires_grad=True):
        return torch.zeros(size=size, dtype=dtype, device=device, requires_grad=requires_grad)
    
    def tensor_raw_ptr(tensor):
        return ctypes.c_void_p(tensor.data_ptr())
"""