import abc

from seastar.compiler.val.val import Val
from ..utils import infer_val_type
from ..program import Stmt

from seastar.compiler.val.val_factory import ValFactory

class Op(abc.ABC):
    def __init__(self, op, fprog):
        self._op = op
        self.fprog = fprog
        self.val_factory = ValFactory()

    def __call__(self, *args, **kargs):
        """Any type/shape inconsistency can be detected by executing the op"""
        assert len(args) > 0, str(self._op) + " received list argument of lenth 0"
        if len(kargs) > 0:
            raise NotImplementedError('Do not support keyword arugmented ops')
        try:
            ret = self._op(*tuple(arg.v for arg in args), **kargs)
        except Exception as e:
            raise e
        if isinstance(ret, tuple) or isinstance(ret, list):
            raise NotImplementedError("Ops that return multiple tensors are not supported op:", str(self), 'ret:', ret)
        else:
            first_backend = args[0].backend
            vtype = infer_val_type(args)
            assert all(val.backend == first_backend for val in args)
            bkend = first_backend
            ret_val = self.val_factory.create(vtype, ret, bkend, None, self.fprog, False)
            def call(*arg_list):
                return self._op(*arg_list, **kargs)
            self.fprog.append_stmt(Stmt.create_stmt(self.to_schema(), 
                                                    args=list((arg.var if isinstance(arg, Val) else arg for arg in args)), 
                                                    ret=ret_val.var,
                                                    callback=call))
        return ret_val
    
    def __str__(self):
        return  str(self._op)
    
    def __repr__(self):
        return str(self)
    
    @abc.abstractmethod
    def to_schema(self):
        """translate backend-specific ops to uniform schema"""

# Currently only supports torch
# class AggOp(abc.ABC):
#     def __call__(self, fprog, args):
#         bkend = args[0].backend
#         vtype = ValType.DEST # Aggregation op are almost always used in forward propagation. This assumption may break in the future.
#         t = args[0].v 
#         val_factory = ValFactory()
#         ret_val = val_factory.create(vtype, t.clone().detach().requires_grad_(False), bkend, None, fprog, False)
#         fprog.append_stmt(Stmt.create_stmt(self.to_schema(),
#                                                 args=list((arg.var if isinstance(arg, Val) else arg for arg in args)),
#                                                 ret = ret_val.var,
#                                                 callback=None))
#         return ret_val

#     @abc.abstractmethod
#     def to_schema(self):
#         """Aggregation op schema"""

# class AggMaxOp(AggOp):
#     def to_schema(self):
#         return Schema('AggMax')

# class AggMinOp(AggOp):
#     def to_schema(self):
#         return Schema('AggMin')

# class AggMeanOp(AggOp):
#     def to_schema(self):
#         return Schema('AggMean')
