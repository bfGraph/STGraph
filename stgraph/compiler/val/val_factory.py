from stgraph.compiler.utils import ValType
from stgraph.compiler.val.pytorch.torch_val import TorchVal

class ValFactory:
    def __init__(self):
        """Factory class to create Val objects"""
        pass

    def create(self, type: ValType, tensor, backend, id, fprog, reduce_dim):
        val_backend = self.get_val_backend(backend)

        return val_backend(
            backend=backend,
            tensor=tensor,
            val_type=type,
            id=id,
            fprog=fprog,
            reduce_dim=reduce_dim,
        )

    def get_val_backend(self, backend):
        key, _ = backend
        if key == "torch":
            return TorchVal
        else:
            raise NotImplementedError(
                f"Backend support for {key} has not been implemented yet"
            )