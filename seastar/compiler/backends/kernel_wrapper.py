class KernelWrapper():
    def __init__(self):
        pass
    
    @staticmethod
    def forward(ctx, executor, kid, kernel_args, rets, *args):
        ctx.backward_cache = executor, kid
        ret = executor.forward_cb(kid, kernel_args, rets, args)
        return ret

    @staticmethod
    def backward(ctx, *gradout):
        executor, kid = ctx.backward_cache
        return (None, None, None, None) + executor.backward_cb(kid, gradout)