import abc

from ..program import Stmt
from ..schema import Schema

from stgraph.compiler.val.val import Val
from stgraph.compiler.utils import ValType
from stgraph.compiler.val.val_factory import ValFactory

class AggOp(abc.ABC):
    def __call__(self, fprog, args):
        bkend = args[0].backend
        vtype = ValType.DEST # Aggregation op are almost always used in forward propagation. This assumption may break in the future.
        t = args[0].v 
        val_factory = ValFactory()
        ret_val = val_factory.create(vtype, t.clone().detach().requires_grad_(False), bkend, None, fprog, False)
        fprog.append_stmt(Stmt.create_stmt(self.to_schema(),
                                                args=list((arg.var if isinstance(arg, Val) else arg for arg in args)),
                                                ret = ret_val.var,
                                                callback=None))
        return ret_val

    @abc.abstractmethod
    def to_schema(self):
        """Aggregation op schema"""

class AggMaxOp(AggOp):
    def to_schema(self):
        return Schema('AggMax')

class AggMinOp(AggOp):
    def to_schema(self):
        return Schema('AggMin')

class AggMeanOp(AggOp):
    def to_schema(self):
        return Schema('AggMean')