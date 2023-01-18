from collections import namedtuple
from ..utils import is_const_scalar, ParallelMode, WriteLocation, WriteType

class KernelContext():
    def __init__(self, unit, index_type):
        self.loaded_args = set()
        self.cur_stmt_ctx = None
        self.offset_cache = {}
        self.unit = unit
        self.src_var_offset_init = "" 
        self.dst_var_offset_init = ""
        self.edge_var_offset_init = ""
        self.param_offset_init = ""
        self.offset_count = 0
        self.offset_prefix = 'offset'
        self.index_type = index_type
        self.init_offset_cache()
        if unit.use_fa_tmpl():
            self.template_name = 'fa'
        else:
            self.template_name = 'v2'

    def init_offset_cache(self):
        for s in self.unit.program:
            for arg in self.kernel_argument_used_in_stmt(s):
                offset_key = self.get_offset_key(arg)
                if offset_key not in self.offset_cache:
                    offset = self.query_offset(arg).strip('[]')
                    offset_id = self.offset_prefix+str(self.offset_count)
                    self.offset_count += 1
                    init_stmt = '{index_type} {offset_id} = {offset};'.format(index_type=self.index_type,
                                                                              offset_id=offset_id,
                                                                              offset=offset)
                    if 'src_id' in offset:
                        self.src_var_offset_init += init_stmt
                    elif 'dst_id' in offset:
                        self.dst_var_offset_init += init_stmt
                    elif 'eid' in offset:
                        self.edge_var_offset_init += init_stmt
                    else:
                        self.param_offset_init += init_stmt
                    self.offset_cache[offset_key] = '[' + offset_id + ']'
        print('offset_cache:', self.offset_cache)

    def get_offset_key(self, var):
        return (tuple(var.var_shape), var.val_type)

    def scalar_var_offset(self, var):
        if var.is_srcvar():
            ret = '[src_id]'
        elif var.is_dstvar():
            ret = '[dst_id]'
        elif var.is_edgevar():
            ret = '[eid]'
        else:
            ret = '[0]'
        return ret

    def vector_var_offset(self, var):
        if var.var_shape[-1] == 1:
            if var.is_srcvar():
                ret = '[src_id*blockDim.y + ty]'
            elif var.is_dstvar():
                ret = '[dst_id*blockDim.y + ty]'
            elif var.is_edgevar():
                ret = '[eid*blockDim.y + ty]'
            else:
                ret = '[ty]'
        else:
            if var.is_srcvar():
                ret = '[src_id*blockDim.x + tx]'
            elif var.is_dstvar():
                ret = '[dst_id*blockDim.x + tx]'
            elif var.is_edgevar():
                ret = '[eid*blockDim.x + tx]'
            else:
                ret = '[tx]'
        return ret

    def matrix_var_offset(self, var):
        if var.is_srcvar():
            ret = '[src_id*blockDim.y*blockDim.x + ty*blockDim.x + tx]'
        elif var.is_dstvar():
            ret = '[dst_id*blockDim.y*blockDim.x + ty*blockDim.x + tx]'
        elif var.is_edgevar():
            ret = '[eid*blockDim.y*blockDim.x + ty*blockDim.x + tx]'
        else:
            ret = '[ty*blockDim.x + tx]'
        return ret
         
    def query_offset(self, var):
        offset_key = self.get_offset_key(var)
        if offset_key not in self.offset_cache:
            '''Assume var_shape is of two dimensions'''
            ret = ''
            if len(var.var_shape) == 2:
                if var.var_shape[-1] == 1:
                    if var.var_shape[-2] == 1:
                        ret = self.scalar_var_offset(var)
                    else:
                        ret = self.vector_var_offset(var)
                elif var.var_shape[-2] == 1:
                    ret = self.vector_var_offset(var)
                else:
                    ret = self.matrix_var_offset(var)
            elif len(var.var_shape) == 1:
                if var.var_shape[-1] == 1:
                    ret = self.scalar_var_offset(var)
                else:
                    ret = self.vector_var_offset(var)
            else:
                raise NotImplementedError('Only support generate code for var shape 1 and 2 not', len(var.var_shape))
            return ret
        else:
            return self.offset_cache[offset_key]
                
    def kernel_argument_used_in_stmt(self, stmt):
        kernel_arguments = set()
        for arg in stmt.args:
            if not is_const_scalar(arg)  and arg not in self.unit.tmps:
                kernel_arguments.add(arg)
        mat_output = True if stmt.ret in self.unit.unit_rets() else False 
        if mat_output:
            kernel_arguments.add(stmt.ret)
        return kernel_arguments
    
    def set_stmt_ctx(self, stmt):
        kernel_arguments = self.kernel_argument_used_in_stmt(stmt)
        write_type = WriteType.NONE
        write_location = WriteLocation.NONE
        mat_output = True if stmt.ret in self.unit.unit_rets() else False 
        if mat_output:
            if self.write_inner(stmt):
                write_location = WriteLocation.INNER
            else:
                write_location = WriteLocation.OUTER
            if stmt.is_agg():
                if write_location == WriteLocation.INNER:
                    write_type = WriteType.ATOMIC 
                else:
                    if self.eq_dim(stmt.ret.var_shape, self.unit.max_dims()):
                        write_type = WriteType.ASSIGN
                    else:
                        write_type = WriteType.ATOMIC 
            else:
                if self.eq_dim(stmt.ret.var_shape, self.unit.max_dims()):
                    write_type = WriteType.ASSIGN
                else:
                    write_type = WriteType.ATOMIC 
        self.cur_stmt_ctx = StmtGenCtx(write_type, write_location, kernel_arguments)
    
    def eq_dim(self, var_shape, dim_list):
        return list(var_shape) == dim_list

    def write_inner(self, stmt):
        ret = False
        if stmt.ret.is_srcvar() and self.unit.parallel_mode() != ParallelMode.SrcParallel:
            ret = True
        elif stmt.ret.is_dstvar() and self.unit.parallel_mode() != ParallelMode.DstParallel:
            ret = True
        elif not stmt.is_agg():
            # Edge-wise op returns edge-wise feature, therefore must be written inner
            ret = True
        return ret

class LinearizedKernelContext(KernelContext):
    def __init__(self, unit, index_type):
        super(LinearizedKernelContext, self).__init__(unit, index_type)
    
    def graph_type_key(self, var):
        if var.is_srcvar():
            return 'src_id'
        elif var.is_dstvar():
            return 'dst_id'
        elif var.is_edgevar():
            return 'eid'
        else:
            return '0'
    
    def query_offset(self, var):
        offset_key = self.get_offset_key(var)
        if offset_key not in self.offset_cache:
            var_dim = 1
            for d in var.var_shape:
                var_dim = d * var_dim 
            unit_dim = 1
            for d in self.unit.max_dims():
                unit_dim = d *unit_dim
            if unit_dim == var_dim:
                # Handling element-wise
                return '[{gid} * {dim} + tx]'.format(gid=self.graph_type_key(var), dim=str(var_dim))
            else:
                # Handling broadcast
                assert unit_dim > var_dim, 'Unit dim must be no smaller than var dim'
                if len(self.unit.max_dims()) == 2:
                    assert var_dim == self.unit.max_dims()[-2], 'Currently the bcast_dim is required to be the second last dimenstion of unit dim.'
                elif len(self.unit.max_dims()) == 1:
                    assert var_dim == 1, 'For scalar max dims, the bcast dim is required to be one'
                else:
                    raise NotImplementedError('3-dimensional feature is not supported yet')
                return '[{gid} * {bcast_dim} + tx/{feature_dim}]'.format(gid=self.graph_type_key(var),
                                                                         bcast_dim=str(var_dim),
                                                                         feature_dim=str(self.unit.max_dims()[-1]))
        else:
            return self.offset_cache[offset_key]

StmtGenCtx = namedtuple('StmtGenCtx', ['write_type', 'write_location', 'kernel_arguments'])
StmtGenCtx.__doc__ = '''
The context for generate current statement. Each statment has several customizable points.

write_type - WriteType. use atomic instructions if the program output has different dimension with current statement's ret val.
write_location - WriteLocation. materialize the stament whithin the edge sequential loop or outside
kernel_arguments -  set(). Returns the set of args and ret that are kernel arguments.
'''