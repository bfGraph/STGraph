from abc import ABC, abstractmethod

class SeastarBackend(ABC):
    def __init__(self):
        self.backend_name = None
        self.backend_module = None
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
    