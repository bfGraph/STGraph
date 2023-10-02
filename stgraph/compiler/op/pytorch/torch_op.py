from stgraph.compiler.op.op import Op
from stgraph.compiler.schema import Schema

class TorchOp(Op):
    def to_schema(self):
        if 'module' in str(type(self._op)):
            return Schema(type(self._op).__name__.split('.')[-1], **{key: val for key, val in self._op.__dict__.items() if not key.startswith('_')})
        elif 'builtin_function_or_method' in str(type(self._op)):
            return Schema(str(self._op).split()[2])
        return self._op.__dict__