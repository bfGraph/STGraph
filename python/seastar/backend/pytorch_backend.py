import torch
import snoop

class KernelWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, executor, kid, kernel_args, rets, *args):
        ctx.backward_cache = executor, kid
        ret = executor.forward_cb(kid, kernel_args, rets, args)
        return ret

    @staticmethod
    def backward(ctx, *gradout):
        executor, kid = ctx.backward_cache
        return (None, None, None, None) +  executor.backward_cb(kid, gradout)
@snoop
def run_egl(executor):
    @snoop
    def new_zeros_call_back(size, dtype, device, requires_grad=True):
        return torch.zeros(size=size, dtype=dtype, device=device, requires_grad=requires_grad)
    @snoop    
    def tensor_raw_ptr(tensor):
        import ctypes
        return ctypes.c_void_p(tensor.data_ptr())
    executor.set_new_zeros_cb(new_zeros_call_back)
    executor.set_raw_ptr_cb(tensor_raw_ptr)
    return executor.execute(KernelWrapper)