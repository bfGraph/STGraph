import math

from .code_gen.cuda_driver import *
from .code_gen.kernel_context import KernelContext, LinearizedKernelContext
from .utils import is_const_scalar, ParallelMode, MAX_THREAD_PER_BLOCK, MAX_BLOCK 

class ExecutionUnit(object):
    unit_count = 0
    def __init__(self, args, tmps, prog, compiled=False):
        self._args = args
        self._tmps = tmps
        self._compiled = compiled
        self._prog = prog
        self._rets = set()
        self._kernel_name = 'K' + str(ExecutionUnit.unit_count)
        self._parallel_mode = None
        self._unit_rets_cached = None
        self._unit_args_cached = None
        self._max_dims_cached = None
        self._parent_units = set()
        if self.feature_size() >= 0:
            self._template_name = 'fa'
        else:
            self._template_name = 'v2'
        ExecutionUnit.unit_count += 1

    def __str__(self):
        return '\n-----------\nparallel_mode:{p_mode}\nargs:{args}\nrets:{rets}\ntmps:{tmps}\ncompiled:{compiled}\nprog:{prog}' \
                    .format(args=str(self._args), rets=str(self._rets), compiled=str(self._compiled), prog=str(self._prog), all_vars=self.get_all_vars(), p_mode=self._parallel_mode, tmps=self._tmps)
    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self._kernel_name)

    def __eq__(self, other):
        return isinstance(other, ExecutionUnit) and self._kernel_name == other._kernel_name
    
    def use_fa_tmpl(self):
        return self._template_name == 'fa'
    
    def create_context(self, index_type):
        if self.use_fa_tmpl():
            return LinearizedKernelContext(self, index_type)
        else:
            return KernelContext(self, index_type)
    
    def set_parallel_mode(self, mode):
        assert isinstance(mode, ParallelMode)
        self._parallel_mode = mode

    def parallel_mode(self):
        return self._parallel_mode
    
    def max_dims(self):
        if not self._max_dims_cached:
            # Assumption: vars are at most two dimensions
            for var in self.get_all_vars():
                shape = var.var_shape
                if len(shape) == 1:
                    if self._max_dims_cached and len(self._max_dims_cached) != 1:
                        #raise NotImplementedError('Var must have consistent feature dimenstions' + str(var) + ' cached ' +str(self._max_dims_cached))
                        continue
                    if not self._max_dims_cached:
                        self._max_dims_cached = [1]
                    self._max_dims_cached = [max(self._max_dims_cached[-1], shape[0])]
                elif len(shape) == 2:
                    if self._max_dims_cached and len(self._max_dims_cached) != 2:
                        #raise NotImplementedError('Var must have consistent feature dimenstions' + str(var) + ' cached ' +str(self._max_dims_cached))
                        continue
                    if not self._max_dims_cached:
                        self._max_dims_cached = [1, 1]
                    self._max_dims_cached = [max(self._max_dims_cached[i], shape[i]) for i in range(2)] 
                else:
                    raise NotImplementedError('Have not suported the case when local var dim larger than 2')
        return self._max_dims_cached
    
    def feature_size(self):
        s = 1
        for d in self.max_dims():
            s = s * d
        return s
    
    def calculate_kernel_params_fa(self, num_nodes):
        feat_size = self.feature_size()
        min_threads = 64
        max_threads = 256
        if feat_size >= min_threads:
            nthrs = min(max_threads, feat_size)
            thrs_per_group = nthrs
            nodes_per_blk = 1
            nblks = num_nodes
        else:
            nthrs = min_threads 
            thrs_per_group = max(1, self.first_pow2_less_than_n(feat_size, nthrs))
            nodes_per_blk = max(2, nthrs/thrs_per_group)
            nblks = (num_nodes+nodes_per_blk-1)//nodes_per_blk
        return int(nblks), int(nthrs), int(thrs_per_group), int(nodes_per_blk)

    def calculate_kernel_params(self, num_nodes):
        kernel_params = self.calculate_kernel_launch_params(num_nodes)
        tile_sizes =  self.compute_tile_sizes(kernel_params[-2], kernel_params[-1])
        return kernel_params, tile_sizes
    
    def first_pow2_less_than_n(self, n, upper_bound):
        while upper_bound > n:
            upper_bound = upper_bound // 2
        return upper_bound


    def calculate_kernel_launch_params(self, num_nodes):
        max_dims = self.max_dims()
        total_dim = 1
        for dim in max_dims:
            total_dim = total_dim*dim
        if total_dim < MAX_THREAD_PER_BLOCK:
            bdim_x = max_dims[-1]
            if len(max_dims) == 1:
                bdim_y = 1
            else:
                bdim_y = max_dims[-2]
            gdim_x = 1
            gdim_y = min(num_nodes, MAX_BLOCK)
        else:
            bdim_x = min(max_dims[-1], 32)
            if len(max_dims) == 1:
                bdim_y = 1
            else:
                bdim_y = min(max_dims[-2], 32)
            gdim_x = int((max_dims[-1] + bdim_x -1) /bdim_x)
            gdim_y = min(num_nodes, MAX_BLOCK)
        return (gdim_x, gdim_y, bdim_x, bdim_y)
    
    def compute_tile_sizes(self, blockDimx, blockDimy):
        WARP_SIZE = 32
        thread_dims = [blockDimx, blockDimy]
        prod = blockDimx * blockDimy
        if prod < 2 * WARP_SIZE:
            return thread_dims# means no tiling at all
        NWARP = prod/WARP_SIZE
        N_Ar = 0
        N_Aw = 0
        N_br = 0
        N_bw = 0
        kernel_args = self.kernel_args()
        for stmt in self._prog:
            for arg in stmt.args:
                if arg in kernel_args:
                    if arg.var_shape == self._max_dims_cached:
                        N_Ar += 1
                    else:
                        N_br += 1
            if stmt.ret in kernel_args:
                if stmt.ret.var_shape == self._max_dims_cached:
                    N_Aw += 1
                else:
                    N_bw += 1
        n =  thread_dims[-1]
        m =  thread_dims[-2]
        cof = N_Ar if n <= WARP_SIZE else WARP_SIZE
        if N_bw > 0:
            x1, x2 = self.nearest_pow2(math.sqrt(cof*n/N_bw))
            t1 = N_Ar * n/x1 + N_bw * x1
            t2 = N_Ar * n/x2 + N_bw * x2
            print('option1:',x1, 'sectors:',t1, 'option2:',x2, 'sectors:', t2)
            tile_sizex = x1 if t1 <= t2 else t2
        else:
            tile_sizex = WARP_SIZE if WARP_SIZE < blockDimx else blockDimx
        tile_sizey = int(WARP_SIZE/tile_sizex)
        assert tile_sizex * tile_sizey == WARP_SIZE
        while tile_sizey > blockDimy:
            tile_sizey = tile_sizey / 2
            tile_sizex = tile_sizex * 2
        print('In compute tiling', N_Ar, N_Aw, N_br, N_bw, self.kernel_args(), 'tx:', tile_sizex, 'ty', tile_sizey)
        return [int(tile_sizex), int(tile_sizey)]
        
    def nearest_pow2(self, targ):
        i = 1
        while 2 * i < targ:
            i = i*2
        diff1 = i - targ
        diff2 = targ - i/2
        return i, i/2
    
    def unit_args(self):
        if not self._unit_args_cached:
            self._unit_args_cached = sorted([arg for arg in self._args if not is_const_scalar(arg)], key=lambda x : x.id)
        return self._unit_args_cached

    def unit_rets(self):
        if not self._unit_rets_cached:
            self._unit_rets_cached = sorted([ret for ret in self._rets if not is_const_scalar(ret)], key=lambda x: x.id)
        return self._unit_rets_cached
    
    def all_rets(self):
        return set([stmt.ret for stmt in self.program])
    
    def kernel_args(self):
        return self.unit_args() + self.unit_rets()
    
    def materilized_vars(self):
        if self.compiled:
            return self._rets.union(self._args)
        else:
            return self.get_all_vars()
    
    def get_all_args(self):
        '''
            return the set of all vars used/returned in the program of this exec unit
        '''
        var_set = set()
        for stmt in self._prog:
            for var in stmt.args:
                if not is_const_scalar(var):
                    var_set.add(var)
        return var_set
    def get_all_vars(self):
        '''
            return the set of all vars used/returned in the program of this exec unit
        '''
        var_set = set()
        for stmt in self._prog:
            for var in stmt.args:
                if not is_const_scalar(var):
                    var_set.add(var)
                var_set.add(stmt.ret)
        return var_set
    
    def add_ret_val(self, ret_val):
        self._rets.add(ret_val)
    
    def max_ret_id(self):
        return sorted([ret.int_id for ret in self.unit_rets()])[-1]

    def prepare_compiled_kernel(self, graph_info, compiled_module):
        if self.parallel_mode() == ParallelMode.DstParallel:
            row_offsets = graph_info.in_row_offsets.data
            col_indices = graph_info.in_col_indices.data
            eids = graph_info.in_eids.data
        else:
            row_offsets = graph_info.out_row_offsets.data
            col_indices = graph_info.out_col_indices.data
            eids = graph_info.out_eids.data
        max_dims = [1, 1]
        if len(self.max_dims()) == 1:
            max_dims[-1] = self.max_dims()[-1]
        elif len(self.max_dims()) == 2:
            max_dims = self.max_dims()
        else:
            raise NotImplementedError('Feature dimension larger than 2 are not supported.')
        num_nodes = graph_info.number_of_nodes
        if self.use_fa_tmpl():
            launch_config = self.calculate_kernel_params_fa(num_nodes)
            print('template name:', self._template_name, 'number of nodes:', num_nodes, 'launch_config', launch_config)
            self._K = FeatureAdaptiveKernel(num_nodes, row_offsets, col_indices, eids, max_dims, self._kernel_name, compiled_module, launch_config)
        else:
            launch_config, tile_sizes = self.calculate_kernel_params(num_nodes)
            print('template name:', self._template_name, 'number of nodes:', num_nodes, 'launch_config', launch_config, 'tile_sizes', tile_sizes, 'max_dims', self.max_dims())
            self._K = V2Kernel(num_nodes, row_offsets, col_indices, eids, max_dims, self._kernel_name, compiled_module, launch_config, tile_sizes)

    def reset_graph_info(self, graph_info):
        if self.parallel_mode() == ParallelMode.DstParallel:
            row_offsets = graph_info.in_row_offsets.data
            col_indices = graph_info.in_col_indices.data
            eids = graph_info.in_eids.data
        else:
            row_offsets = graph_info.out_row_offsets.data
            col_indices = graph_info.out_col_indices.data
            eids = graph_info.out_eids.data
        self._K.reset_graph_info(graph_info.number_of_nodes, row_offsets, col_indices, eids)

    def kernel_run(self, tensor_list):
        assert self._K, 'Must call prepare_compiled_kernel before call kernel_run.'
        self._K.run(tensor_list)
    
    def merge_with_independent_unit(self, other):
        # union their inputs and outputs
        self._unit_args_cached = None
        self._unit_rets_cached = None
        self._max_dims_cached = None
        self._args = self._args.union(other._args)
        self._rets = self._rets.union(other._rets)
        self._tmps = self._tmps.union(other._tmps)
        # merge stmts
        first_agg = None
        for s in self._prog:
            if s.is_agg():
                first_agg = s
                break
        self._prog.insert_stmts_before(first_agg, list(other._prog))
        # adjust parallel mode
        dst_parallel_count = 0
        for s in  self._prog:
            if s.is_agg():
                if s.ret.is_dstvar():
                    dst_parallel_count += 1
                else:
                    dst_parallel_count -= 1
        if dst_parallel_count >= 0:
            self._parallel_mode = ParallelMode.DstParallel
        else:
            self._parallel_mode = ParallelMode.SrcParallel
        return self
    
    def depends_on(self, other):
        if self == other or self.is_child_of(other):
            return True
        else:
            return any([u.depends_on(other) for u in self._parent_units])

    def add_parent_unit(self, parent):
        self._parent_units.add(parent)
    
    def has_parent(self) :
        return len(self._parent_units) > 0

    def is_child_of(self, other):
        return other in self._parent_units

    @property
    def program(self):
        return self._prog

    @property
    def tmps(self):
        return self._tmps

    @property
    def compiled(self):
        return self._compiled

    @property
    def kernel_name(self):
        return self._kernel_name

class Kernel():
    def reset_graph_info(self, number_of_nodes, row_offsets, col_indices, eids):
        self.const_kernel_args[0] = row_offsets
        self.const_kernel_args[1] = eids
        self.const_kernel_args[2] = col_indices
        self.const_kernel_args[3] = c_int(number_of_nodes)
        for i in range(4):
            self.const_kernel_ptrs[i] = c_void_p(addressof(self.const_kernel_args[i]))

    def run(self, tensor_list):
        try:
            kernel_ptrs = [c_void_p(addressof(arg)) for arg in tensor_list] + self.const_kernel_ptrs
            params =  (c_void_p * len(kernel_ptrs))(*kernel_ptrs)
            ret = cuLaunchKernel(self.K, 
                                 self.launch_config[0], 
                                 self.launch_config[1],
                                 self.launch_config[2], 
                                 self.launch_config[3],
                                 self.launch_config[4],
                                 self.launch_config[5],
                                 0, None, params, None)
            if ret:
                raise Exception('cuLaunchKernel', ret)
        except Exception as e:
            raise e

class V2Kernel(Kernel):
    def __init__(self, num_nodes, row_offsets, col_indices, eids, max_dims, kernel_name, compiled_module, launch_config, tile_sizes):
        self.scalar_args = [c_int(num_nodes), c_int(max_dims[1]), c_int(max_dims[0]), c_int(tile_sizes[0]), c_int(tile_sizes[1])]
        self.const_kernel_args =  [row_offsets, eids, col_indices] + self.scalar_args
        self.const_kernel_ptrs = [c_void_p(addressof(v)) for v in self.const_kernel_args]
        self.K = c_void_p(0)
        ret = cuModuleGetFunction(byref(self.K), compiled_module, c_char_p(kernel_name.encode()))
        if ret:
            raise Exception('cuModuleGetFunction', ret)
        self.launch_config = launch_config[0],launch_config[1], 1, launch_config[2], launch_config[3],1

class FeatureAdaptiveKernel(Kernel):
    def __init__(self, num_nodes, row_offsets, col_indices, eids, max_dims, kernel_name, compiled_module, launch_config):
        self.scalar_args = [c_int(num_nodes), c_int(max_dims[1]), c_int(max_dims[0]), c_int(launch_config[2]), c_int(launch_config[3])]
        self.const_kernel_args =  [row_offsets, eids, col_indices] + self.scalar_args
        self.const_kernel_ptrs = [c_void_p(addressof(v)) for v in self.const_kernel_args]
        self.K = c_void_p(0)
        ret = cuModuleGetFunction(byref(self.K), compiled_module, c_char_p(kernel_name.encode()))
        if ret:
            raise Exception('cuModuleGetFunction', ret)
        self.launch_config = launch_config[0],1,1,launch_config[1],1,1