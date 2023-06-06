import traceback
import abc
from collections.abc import Iterable

from .val import Val
from .utils import infer_val_type, ValType
from .schema import Schema
from .program import Stmt

from seastar.compiler.val import ValCreator

def create_op(op, backend, fprog):
    if backend == 'torch':
        return TorchOp(op, fprog)
    else:
        raise NotImplementedError(backend +' is not supported yet')

class Op(abc.ABC):
    def __init__(self, op, fprog):
        self._op = op
        self.fprog = fprog
        self.val_creator = ValCreator()

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
            ret_val = self.val_creator.create(vtype, ret, bkend, None, self.fprog, False)
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

class TorchOp(Op):
    def to_schema(self):
        if 'module' in str(type(self._op)):
            return Schema(type(self._op).__name__.split('.')[-1], **{key: val for key, val in self._op.__dict__.items() if not key.startswith('_')})
        elif 'builtin_function_or_method' in str(type(self._op)):
            return Schema(str(self._op).split()[2])
        return self._op.__dict__

# Currently only supports torch
class AggOp(abc.ABC):
    def __call__(self, fprog, args):
        bkend = args[0].backend
        vtype = ValType.DEST # Aggregation op are almost always used in forward propagation. This assumption may break in the future.
        t = args[0].v 
        val_creator = ValCreator()
        ret_val = val_creator.create(vtype, t.clone().detach().requires_grad_(False), bkend, None, fprog, False)
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
