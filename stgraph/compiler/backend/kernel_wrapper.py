class KernelWrapper():
    def __init__(self):
        pass
    
    @staticmethod
    def forward(executor, kid, kernel_args, rets, *args):
        ret = executor.forward_cb(kid, kernel_args, rets, args)
        return ret

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Saving executor and kid in backward cache
        ctx.backward_cache = inputs[0], inputs[1]
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, *gradout):
        executor, kid = ctx.backward_cache
        return (None, None, None, None) + executor.backward_cb(kid, gradout)