import abc
import sys, inspect
from collections import namedtuple
from .utils import ValType,is_const_scalar, WriteType, WriteLocation, infer_val_type
from .schema import Schema

GradInfo = namedtuple('GradInfo', ['targ', 'args', 'grad_x', 'op_schema'])

impl_registry = {}
cb_registry = {}
TMP_SUFFIX ='_tmp'

def register_or_look_up_backend_cb(stmt, cb):
    op_name = stmt.op_name.lower()
    if cb:
        if op_name not in cb_registry:
            cb_registry[op_name] = cb 
        return cb
    else:
        if op_name in cb_registry:
            return cb_registry[op_name] 

def look_up_registry(stmt):
    op_name = stmt.op_name.lower()
    op_type = stmt.op_type
    op_impl = None
    if op_name not in impl_registry:
        # We don't necessarily generate ops for Node-wise op as they can be 
        # supported by backends
        if stmt.is_edgewise() and op_name not in cb_registry:
            print('Warnning: EdgeType op' + op_name + ' is not registered in any registry!')
    else:
        op_impl = impl_registry[op_name]
    return op_impl

class OpImpl(abc.ABC):
    '''New ops need to inherit from this class with name "XXXOp"'''
    def __init__(self, fstmt, create_var_cb, create_stmt_cb):
        self.fstmt = fstmt
        self.create_var = create_var_cb
        self.create_stmt = create_stmt_cb
    
    def grad(self, y, grad_y):
        ret_list = []
        for pos, x in enumerate(self.fstmt.args):
            if not is_const_scalar(x) and x.requires_grad:
                ret_stmts = self.grad_impl(pos, x, y, grad_y)
                ret_list.append((x, ret_stmts))
        return ret_list

    def gen_var(self, var, ctx):
        kctx = ctx
        ctx = ctx.cur_stmt_ctx
        if is_const_scalar(var):
            return str(var)
        if var == self.ret:
            prefix = '' if 'agg' in var.stmt.op_name.lower() else var.dtype_str  + ' '
            return  prefix  + var.id + TMP_SUFFIX
        else:
            if var in ctx.kernel_arguments:
                return var.id + kctx.query_offset(var)
            else:
                return var.id + TMP_SUFFIX
    
    def gen_write(self, ctx):
        kctx =  ctx
        ctx = kctx.cur_stmt_ctx
        if ctx.write_type == WriteType.NONE:
            return ('inner_write', '')
        val = ''
        var = self.ret.id + kctx.query_offset(self.ret) 
        delta = self.ret.id + TMP_SUFFIX 
        if ctx.write_type == WriteType.ADD:
            val = '{var} += {delta};'.format(var=var, delta=delta)
        elif ctx.write_type == WriteType.ATOMIC:
            var_split = var.split('[')
            # replacing var[offset] with var + offset
            new_var = var_split[0]  + '+' + var_split[1][:-1] 
            divisor=''
            if 'sum' in self.fstmt.op_name.lower():
                op = 'Add' 
            elif 'max' in self.fstmt.op_name.lower():
                op = 'Max'
            elif 'min' in self.fstmt.op_name.lower():
                op = 'Min'
            elif 'mean' in self.fstmt.op_name.lower():
                op = 'Add'
                divisor='/(end_off-start_off)'
                if ctx.write_location != WriteLocation.OUTER:
                  raise NotImplementedError('Cannot support innter write of mean result due to unknown number of edges')
            else:
                raise NotImplementedError('Atomic instruction for', self.fstmt.op_name, 'is not implemented')
            val = 'atomic{op}({var}, {delta}{divisor});'.format(var=new_var, delta=delta, op=op, divisor=divisor)
        elif ctx.write_type == WriteType.ASSIGN:
            val = '{var} = {delta};'.format(var=var, delta=delta)

        key = 'inner_write'
        if ctx.write_location == WriteLocation.OUTER:
            key = 'outter_write' 
        return key, val
    
    def gen_load(self, ctx):
        k='load'
        v=''
        for arg in self.args:
            if arg in ctx.cur_stmt_ctx.kernel_arguments and arg not in ctx.loaded_args:
                v += '{type} {var_tmp} = {var}; '.format(type=arg.dtype_str, var_tmp=arg.id+TMP_SUFFIX, var=arg.id+ctx.query_offset(arg))
                ctx.loaded_args.add(arg)
        return k, v.strip(' ')
    
    def gen_edge_info_map(self, ctx):
        m = {'compute':'', 'load':''}
        key,val = self.gen_write(ctx)
        m[key] = val
        #k,v = self.gen_load(ctx)
        #m[k] = v
        return m
    
    def gen_agg_info_map(self, ctx):
        m = {'init':'', 'compute':'', 'inner_write':'', 'outter_write':''}
        key,val = self.gen_write(ctx)
        m[key] = val
        return m
    

    def create_var_like(self, x):
        return self.create_var(var_shape=x.var_shape,
                               var_dtype=x.var_dtype,
                               val_type=x.val_type,
                               device=x.device)
    
    def multiply_grad(self, dzdy, dydx, x):
        dim_size = len(dzdy.var_shape)
        dim_sizex = len(x.var_shape)
        assert dim_size == dim_sizex
        max_dim = [1 for i in range(dim_size)]
        if is_const_scalar(dydx):
            max_dim = dzdy.var_shape
        else:
            for i in range(dim_size):
                max_dim[i] = max(dzdy.var_shape[i], dydx.var_shape[i])
        ret = [self.create_stmt(Schema('Mul'), args=[dzdy, dydx], ret=self.create_var(var_shape=max_dim,
                                                                                       var_dtype=dzdy.var_dtype,
                                                                                       val_type=infer_val_type([dzdy, dydx]),
                                                                                       device=dzdy.device))]
        if x.var_shape != max_dim:
            print("Inconsistent shape between x and its gradient")
            if len(x.var_shape) != len(max_dim):
                raise NotImplementedError('Multiply grad has not supported input and gradient that have different dimension')
            diff_dim = -1
            diff_count = 0
            for i in range(len(max_dim)):
                if max_dim[i] != x.var_shape[i]:
                    diff_dim = i
                    diff_count += 1
            if diff_count > 1:
                raise NotImplementedError('Multiply grad has not supported input and gradient that have more than 1 different dims')
            ret.append(self.create_stmt(Schema('Sum', dim=diff_dim, keep_dim=True), args=[ret[-1].ret], ret=self.create_var_like(x)))
        return ret

    @abc.abstractmethod
    def grad_impl(self, pos, x, y, grad_y):
        '''return a map with keys: args, grad_x and op_schema'''

    @abc.abstractmethod
    def gen_code(self, ctx):
        '''return the cuda code that corresponding to this op'''
    
    @property
    def args(self):
        return self.fstmt.args
    
    @property
    def ret(self):
        return self.fstmt.ret

    @property
    def op_schema(self):
        return self.fstmt.op_schema

class BinaryOpImpl(OpImpl):
    def grad_impl(self, pos, x, y, grad_y):
        stmt_list = self._grad_impl(pos, x, y, grad_y)
        if x.is_nodevar() and self.fstmt.is_edgewise():
            stmt = self.create_stmt(Schema('AggSum'), args=[stmt_list[-1].ret], ret=self.create_var_like(x))
            stmt_list.append(stmt)
        return stmt_list

    @abc.abstractmethod
    def _grad_impl(self, pos, x, y, grad_y):
        '''Opreator specifc implementation of binaryop'''


class AddOp(BinaryOpImpl):
    def _grad_impl(self, pos, x, y, grad_y):
        '''y = x + k => dydx = 1'''
        return self.multiply_grad(dzdy=grad_y, dydx=1, x=x)
                
    def gen_code(self, ctx):
        assert len(self.args) == 2
        val0 = self.gen_var(self.args[0], ctx)
        val1 = self.gen_var(self.args[1], ctx)
        ret = self.gen_var(self.ret, ctx)
        gen_info = self.gen_edge_info_map(ctx)
        gen_info['compute'] = '{ret} = {val0} + {val1};'.format(ret=ret, val0=val0, val1=val1)
        return gen_info
               

class LeakyReluOp(OpImpl):
    def grad_impl(self, pos, x, y, grad_y):
        '''y = leaky_relu(x) => dydx = backward_leaky_relu(x)'''
        stmt_list = []
        var1 = self.create_var_like(x)
        stmt_list.append(self.create_stmt(Schema('BackwardLeakyRelu', **self.op_schema._params), args=[x], ret=var1))
        stmt_list += self.multiply_grad(grad_y, var1, x)
        return stmt_list

    def gen_code(self, ctx):
        arg = self.gen_var(self.args[0], ctx)
        ret = self.gen_var(self.ret, ctx)
        gen_info = self.gen_edge_info_map(ctx)
        gen_info['compute'] = '{ret}={val}>0?{val}:{slope}*{val};'.format(ret=ret,val=arg,slope=self.op_schema._params['negative_slope'])
        return gen_info
               

class ExpOp(OpImpl):
    def grad_impl(self, pos, x, y, grad_y):
        '''y = exp(x) => dydx = exp(x) = y'''
        return self.multiply_grad(dzdy=grad_y, dydx=y, x=x)
    
    def gen_code(self, ctx):
        arg = self.gen_var(self.args[0], ctx)
        ret = self.gen_var(self.ret, ctx)
        gen_info = self.gen_edge_info_map(ctx)
        gen_info['compute'] = '{ret} = exp({val});'.format(ret=ret, val=arg)
        return gen_info
                            

class MulOp(BinaryOpImpl):
    def _grad_impl(self, pos, x, y, grad_y):
        ''' y=x[0]*x[1] => dydx0 = x[1], dydx1 = x[0]'''
        assert pos < 2, 'Mul dealing with two operands'
        return self.multiply_grad(dzdy=grad_y, dydx=self.args[1-pos], x=x)

    def gen_code(self, ctx):
        val0 = self.gen_var(self.args[0], ctx)
        val1 = self.gen_var(self.args[1], ctx)
        ret = self.gen_var(self.ret, ctx)
        gen_info = self.gen_edge_info_map(ctx)
        gen_info['compute'] =  '{ret} = {val0}*{val1};'.format(ret=ret,val0=val0, val1=val1)
        return gen_info

class AggSumOp(OpImpl):
    def grad_impl(self, pos, x, y, grad_y):
        '''y = AggSum(x) => dydx = Bcast(x)'''
        grad_stmt_list = self.multiply_grad(dzdy=grad_y, dydx=1, x=x)
        if not x.is_edgevar():
            last_stmt = grad_stmt_list[-1]
            grad_stmt_list.append(self.create_stmt(Schema('AggSum'), args=[last_stmt.ret], ret=self.create_var_like(x)))
        return grad_stmt_list

    def gen_init(self, var):
        key = 'init'
        val = var.dtype_str + ' '+ var.id + TMP_SUFFIX  + ' = 0;'
        return key, val

    def gen_code(self, ctx):
        val0 = self.gen_var(self.args[0], ctx)
        ret = self.gen_var(self.ret, ctx)
        initk,initv = self.gen_init(self.ret)
        gen_info =self.gen_agg_info_map(ctx)
        if ctx.cur_stmt_ctx.write_location == WriteLocation.INNER:
            gen_info['compute'] = '{ret} = {val};'.format(ret=ret, val=val0)
        else:
            gen_info['compute'] = '{ret} += {val};'.format(ret=ret, val=val0)
        gen_info[initk] = initv
        return gen_info

class AggMaxOp(OpImpl):
    def grad_impl(self, pos, x, y, grad_y):
        '''y = AggMax(x) => dydx = (x)'''
        grad_stmt_list = []
        # More precisely, the type of ret should be of type.E but it's OK as long as we don't materialize it.
        ret = self.create_var_like(x)
        ret._val_type = ValType.E
        grad_stmt_list.append(self.create_stmt(Schema('BackwardAMax'), args=[x, y], ret=ret))
        grad_stmt_list += self.multiply_grad(dzdy=grad_y, dydx=grad_stmt_list[-1].ret, x=x)
        if not x.is_edgevar():
            last_stmt = grad_stmt_list[-1]
            grad_stmt_list.append(self.create_stmt(Schema('AggSum'), args=[last_stmt.ret], ret=self.create_var_like(x)))
        return grad_stmt_list

    def gen_init(self, var):
        key = 'init'
        val = var.dtype_str + ' '+ var.id + TMP_SUFFIX  + ' = 0xff800000;' # 0x7f800000 for inf. 0xff800000 for -inf
        return key, val

    def gen_code(self, ctx):
        val0 = self.gen_var(self.args[0], ctx)
        ret = self.gen_var(self.ret, ctx)
        initk,initv = self.gen_init(self.ret)
        gen_info =self.gen_agg_info_map(ctx)
        if ctx.cur_stmt_ctx.write_location == WriteLocation.INNER:
            gen_info['compute'] = '{ret} = {val};'.format(ret=ret, val=val0)
        else:
            gen_info['compute'] = '{ret} = max({val}, {ret});'.format(ret=ret, val=val0)
        gen_info[initk] = initv
        return gen_info

class BackwardAMaxOp(OpImpl):
    def grad_impl(self, pos, x, y, grad_y):
        '''''' 
        raise NotImplementedError('Grad of grad is not supported')

    def gen_code(self, ctx):
        forward_x = self.gen_var(self.args[0], ctx)
        forward_y = self.gen_var(self.args[1], ctx)
        ret = self.gen_var(self.ret, ctx)
        gen_info = self.gen_edge_info_map(ctx)
        gen_info['compute'] = '{ret} = {forward_x} == {forward_y} ? 1 : 0;'.format(ret=ret, forward_x=forward_x, forward_y=forward_y)
        return gen_info

class TrueDivOp(BinaryOpImpl):
    def _grad_impl(self, pos, x, y, grad_y):
        ''' y = x[0]/x[1] => dydx0 = 1/x[1] dydx1 = BackwardTrueDiv(x[0], x[1])'''
        assert pos < 2, 'TrueDiv dealing with two operands'
        stmt_list = []
        if pos == 0:
            var = self.create_var_like(self.args[1])
            stmt_list.append(self.create_stmt(Schema('TrueDiv'), args=[1, self.args[1]], ret=var))
        else:
            stmt_list.append(self.create_stmt(Schema('Mul'), args=[self.args[1], self.args[1]], ret=self.create_var_like(self.args[1])))
            stmt_list.append(self.create_stmt(Schema('Mul'), args=[-1, self.args[0]], ret=self.create_var_like(self.args[0])))
            var = self.create_var_like(y)
            stmt_list.append(self.create_stmt(Schema('TrueDiv'), args=[stmt_list[-1].ret, stmt_list[-2].ret], ret=var))
        stmt_list += self.multiply_grad(dzdy=grad_y, dydx=var, x=x)
        return stmt_list

    def gen_code(self, ctx):
        left = self.gen_var(self.args[0], ctx)
        right = self.gen_var(self.args[1], ctx)
        ret = self.gen_var(self.ret, ctx)
        gen_info = self.gen_edge_info_map(ctx)
        gen_info['compute'] = '{ret} = {left}/{right};'.format(ret=ret, left=left, right=right)
        return gen_info

class ReluOp(OpImpl):
    def grad_impl(self, pos, x, y, grad_y):
        assert x.val_type == grad_y.val_type
        ret = self.create_var_like(x)
        stmt_list = [self.create_stmt(Schema('BackwardRelu'), args=[x, grad_y], ret=ret)]
        return stmt_list

    def gen_code(self, ctx):
        inp = self.gen_var(self.args[0], ctx)
        ret = self.gen_var(self.ret, ctx)
        gen_info = self.gen_edge_info_map(ctx)
        gen_info['compute'] = '{ret} = {inp} > 0 ? {inp} : 0;'.format(ret=ret, inp=inp)
        return gen_info

class BackwardReluOp(OpImpl):
    def grad_impl(self, pos, x, y, grad_y):
        raise NotImplementedError('Grad for BackwardRelu is not implemented')

    def gen_code(self, ctx):
        assert len(self.args) == 2, 'backward relu takes two arguments but {n} are given'.format(len(self.args))
        inp0 = self.gen_var(self.args[0], ctx)
        inp1 = self.gen_var(self.args[1], ctx)
        ret = self.gen_var(self.ret, ctx)
        gen_info  = self.gen_edge_info_map(ctx)
        gen_info['compute'] = '{ret} = {inp0} > 0 ? {inp1} : 0;'.format(ret=ret, inp0=inp0, inp1=inp1)
        return gen_info

class GTypeCastOp(OpImpl):
    def grad_impl(self, pos, x, y, grad_y):
        raise NotImplementedError('Grad for GTypeCast is not implemented')

    def gen_code(self, ctx):
        raise NotImplementedError('Cannot generate code for GTypeCast')

class BackwardLeakyReluOp(OpImpl):
    def grad_impl(self, pos, x, y, grad_y):
        raise NotImplementedError('Grad of grad is not supported')

    def gen_code(self, ctx):
        x = self.gen_var(self.args[0], ctx)
        ret = self.gen_var(self.ret, ctx)
        gen_info = self.gen_edge_info_map(ctx)
        gen_info['compute'] = '{ret} = {x}>0?1:{slope};'.format(ret=ret,x=x,slope=self.op_schema._params['negative_slope'])
        return gen_info

def register_ops():
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and name.endswith('Op'):
            key = name.split('Op')[0].lower()
            print('Registering', name, 'with key', key)
            impl_registry[key] = obj

register_ops()
