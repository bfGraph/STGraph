from enum import Enum
from collections.abc import Iterable
unused_ids = set()
val_seq = 0
MAX_THREAD_PER_BLOCK=1024
MAX_BLOCK=65535
var_prefix='V'
cen_attr_postfix='cen'
inb_attr_postfix='inb'

class EdgeDirection(Enum):
    IN = 0
    OUT = 1

class ValType(Enum):
    S = 0
    D = 1
    E = 2
    P = 3

class OpType(Enum):
    S = 0
    E = 1
    A = 2
    D = 3

class FusionType(Enum):
    NEAN = 0
    NN = 1
    NOT_FUSIBLE = 2

class ParallelMode(Enum):
    SrcParallel = 0
    DstParallel = 1

class WriteType(Enum):
    ADD = 0
    ATOMIC = 1
    ASSIGN = 2
    NONE = 3

class WriteLocation(Enum):
    INNER = 0
    OUTER = 1
    NONE = 2

def is_const_scalar(val):
    return type(val) in (str, int, float, bool)

def infer_val_type(vals):
    '''
        When type of vals are different, we return edge type.
        P type is considered same with every one.
    '''
    assert isinstance(vals, Iterable), 'vals must be iterable'
    assert len(vals) >= 1
    for i in range(len(vals)):
        if not is_const_scalar(vals[i]) and vals[i].val_type != ValType.P:
            first_non_p_type = vals[i].val_type
    diff_val_type = any(val.val_type != first_non_p_type for val in vals if not is_const_scalar(val) and val.val_type != ValType.P )
    if diff_val_type:
        vtype = ValType.E
    else:
        vtype = first_non_p_type
    return vtype

def infer_op_type_from_args(op_schema, args):
    if 'agg' in op_schema._op_name.lower():
        return OpType.A
    inf_val_type = infer_val_type(args)
    if inf_val_type == ValType.E:
        return OpType.E
    elif inf_val_type == ValType.S:
        return OpType.S
    elif inf_val_type == ValType.D:
        return OpType.D


def any_var(var_list):
    first_var = None
    for var in var_list:
        if not is_const_scalar(var):
            first_var = var
    return first_var

def bcast_dim(var_list):
    first_var = any_var(var_list)
    assert first_var != None
    maxdim = [dim for dim in first_var.var_shape]
    for i in range(0, len(var_list)):
        if is_const_scalar(var_list[i]):
            continue
        assert len(maxdim) == len(var_list[i].var_shape), str(maxdim) + str(var_list[i].var_shape) + str(var_list)
        for j in range(len(maxdim)):
            maxdim[j] = max(maxdim[j], var_list[i].var_shape[j])
    return maxdim
