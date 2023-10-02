from stgraph.compiler.op.pytorch.torch_op import TorchOp

class OpFactory:
    def __init__(self):
        """ Factory class to create Op objects for different backends"""
        pass
    
    def create(self, op, backend, fprog):
        op_backend = self._get_op_backend(backend)
        return op_backend(op, fprog)
    
    def _get_op_backend(self, backend):
        if backend == "torch":
            return TorchOp
        else:
            raise NotImplementedError(backend +' is not supported yet for Op')