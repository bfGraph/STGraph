import functools
from collections import defaultdict, namedtuple
from collections.abc import Iterable

from .node import CentralNode
from .val import create_edge_val, create_src_node_val, create_dst_node_val, create_param_val
from .op import create_op, AggMaxOp, AggMinOp, AggMeanOp
from .program import Var, Stmt, Program
from .passes import optimize, CF, fuse, visualize
from .schema import Schema
from .autodiff import diff
from .code_gen import code_gen 
from .executor import Executor
from .utils import var_prefix, cen_attr_postfix, inb_attr_postfix

class Context():
    GraphInfo = namedtuple('GraphInfo', ['number_of_nodes',
                                         'number_of_edges',
                                         'in_row_offsets',
                                         'in_col_indices',
                                         'in_eids',
                                         'out_row_offsets',
                                         'out_col_indices',
                                         'out_eids',
                                         'nbits'])
    def __init__(self, func, nspace, run_cb):
        functools.update_wrapper(self, func)
        self._f = func
        self._nspace = nspace
        self._entry_count = 0
        self._run_cb = run_cb
        # Hold reference to parameters of current module to avoid repeated lookup
        self._input_cache = {}
        self._graph_info_cache = None
        self._executor_cache = None

    def __call__(self, **kwargs):
        executor = self._setup_executor(**kwargs)
        ret = self._run_cb(executor)
        if len(ret) == 1:
            return ret[0]
        return ret

    def _setup_executor(self, **kwargs):
        graph = kwargs.get('g', None)
        node_feats = kwargs.get('n_feats', {})
        edge_feats = kwargs.get('e_feats', {})
        if not graph:
            raise NameError('Need to provide the graph as one of keyward arguments')
        graph_info, need_reset = self._update_graph_info(graph)
        if self._entry_count == 0:
            fprog = Program()
            ret = self._trace(node_feats, edge_feats, self._input_cache, fprog)
            print('TracedProgram' + str(fprog), 'Ret value:', ret)
            self._executor_cache = self._diff_then_compile(ret, fprog, graph_info)
        for k, v in node_feats.items():
            self._input_cache[var_prefix + k + cen_attr_postfix] = v
            self._input_cache[var_prefix + k + inb_attr_postfix] = v
        for k, v in edge_feats.items():
            self._input_cache[var_prefix+k] = v
        self._executor_cache.restart(self._input_cache, graph_info if need_reset else None)
        self._entry_count += 1
        return self._executor_cache
        
    def _update_graph_info(self, graph):
        reset = False
        if not (self._graph_info_cache != None
                and self._graph_info_cache.number_of_nodes == graph.number_of_nodes()
                and self._graph_info_cache.number_of_edges == graph.number_of_edges()):
            in_csr = graph.get_in_csr()
            out_csr = graph.get_out_csr()
            self._graph_info_cache = Context.GraphInfo(
                graph.number_of_nodes(),
                graph.number_of_edges(),
                in_csr(0).copy_to_gpu(0),
                in_csr(1).copy_to_gpu(0),
                in_csr(2).copy_to_gpu(0),
                out_csr(0).copy_to_gpu(0),
                out_csr(1).copy_to_gpu(0),
                out_csr(2).copy_to_gpu(0),
                graph.nbits())
            reset = True
        return self._graph_info_cache, reset

    def _trace(self, nfeats, efeats, input_cache, fprog):
        backend = self.find_backend(self._nspace)
        central_node = self._init_central_node(nfeats, efeats, fprog, backend)
        old_libs = defaultdict(dict)
        self._monkey_patch_namespace(old_libs, input_cache, fprog, backend)
        ret = self._f(central_node)
        self._remove_patch(old_libs, backend)
        if ret == None:
            raise NameError('Ret is none. Execution is aborted')
        return [ret.var] if not isinstance(ret, Iterable) else ret.var

    def _diff_then_compile(self, out_set, fprog, graph):
        optimize(fprog)
        vars = []
        for var in out_set:
            vars.append(var)
        forward_exe_units = fuse([fprog], vars)
        grads = []
        for var in vars:
            grads.append(Var.create_var(var_shape=var.var_shape, var_dtype=var.var_dtype, val_type=var.val_type, device=var.device))
        backward_exe_units = diff(vars, grads, forward_exe_units, fprog)
        #visualize.plot_exec_units(forward_exe_units + backward_exe_units)
        compiled_module = code_gen.gen_code(forward_exe_units + backward_exe_units, 'int' if graph.nbits == 32 else 'long long int')
        return Executor(graph, forward_exe_units, backward_exe_units, compiled_module, vars)
        
    def _init_central_node(self, nfeats, efeats, fprog, backend):
        cen = CentralNode()
        if nfeats:
            for k, v in nfeats.items():
                setattr(cen, k, create_dst_node_val(v, backend, id=k+cen_attr_postfix, fprog=fprog))
                for n in cen.innbs:
                    setattr(n, k, create_src_node_val(v, backend, id=k+inb_attr_postfix, fprog=fprog))
        if efeats:
            for k, v in efeats.items():
                for e in cen.inedges:
                    setattr(e, k, create_edge_val(v, backend, id=k, fprog=fprog))
        return cen
    
    def _monkey_patch_namespace(self, old_libs, input_cache, fprog, backend):
        """Symbolizing central node and its innbs and inedges""" 
        if backend[0] == 'torch':
            for i, nspace in enumerate(self._nspace):
                if '__name__' in nspace.__dict__:
                    # symbolizing functions for torch namespace
                    k = self._mapping_key(i, 'function')
                    for key in nspace.__dict__:
                        m = nspace.__dict__[key]
                        if 'function' in str(type(m)):
                            if key in old_libs[k]:
                                raise KeyError('Found', key, ' already in old_libs')
                            old_libs[k][key] = m
                            nspace.__dict__[key] = create_op(m, backend[0], fprog=fprog)
                else:
                    ## Dealing with module
                    for key in nspace.__dict__.keys():
                        # symbolizing parameters for self namespace
                        m = nspace.__dict__[key]
                        k = self._mapping_key(i, key)
                        if key.startswith('_parameters'):
                            for mkey in m.keys():
                                if mkey in old_libs[k]:
                                    raise KeyError('Found', key, ' already in old_libs')
                                old_libs[k][mkey] = m[mkey] 
                                input_cache[var_prefix+mkey] = m[mkey]
                                m[mkey] = create_param_val(m[mkey], backend, id=mkey, fprog=fprog)

                        # symbolizing buffers for self namespace
                        if key.startswith('_buffers'):
                            for mkey in m.keys():
                                if mkey in old_libs[k]:
                                    raise KeyError('Found', key, 'already in old_libs')
                                old_libs[k][mkey] = m[mkey]
                                input_cache[var_prefix+mkey] = m[mkey]
                                m[mkey] = create_param_val(m[mkey], backend, id=mkey, fprog=fprog)

                        # symbolizing modules for self namespace
                        if key.startswith('_modules'):
                            for mkey in m.keys():
                                if mkey in old_libs[k]:
                                    raise KeyError('Found', key, ' already in old_libs')
                                old_libs[k][mkey] = m[mkey]
                                m[mkey] = create_op(m[mkey], backend[0], fprog=fprog)
        else:
            raise NotImplementedError('Backend ' + backend[0] + ' is not supported yet!') 

    def _remove_patch(self, old_libs, backend):
        if backend[0] == 'torch':
            for i, nspace in enumerate(self._nspace):
                if '__name__' in nspace.__dict__:
                    # desymbolizing functions for torch namespace
                    if 'torch' in nspace.__name__.lower():
                        k = self._mapping_key(i, 'function')
                        for key in old_libs[k]:
                            nspace.__dict__[key] = old_libs[k][key]
                else:
                    for key in nspace.__dict__.keys():
                        # desymbolizing parameters for self namespace
                        m = nspace.__dict__[key]
                        k = self._mapping_key(i, key) 
                        if key.startswith('_parameters') or key.startswith('_modules') or key.startswith('_buffers'):
                            for mkey in old_libs[k]:
                                m[mkey] = old_libs[k][mkey]
        else:
            raise NotImplementedError('Backend ' + backend[0] + ' is not supported yet!') 


    def find_backend(self, namespace):
        k =  '__name__'
        for n in namespace:
            if k in n.__dict__:
                name = n.__dict__[k].lower()
                if 'torch' in name:
                    return ('torch', n)
                elif 'tensorflow' in name:
                    return ('tensorflow', n)
                elif 'mxnet' in name:
                    return ('mxnet', n)
                else:
                    raise NotImplementedError("Backend support for " + name + " is not implemnted yet")

    def _mapping_key(self, name_space_id, original_key):
        return str(name_space_id) + str(original_key)


class CtxManager():
    def __init__(self, run_cb):
        self._ctx_map = {}
        self._run_cb = run_cb
    
    def zoomIn(self, nspace, hetero_graph=False):
        def wrapper(func):
            if not func.__name__ in self._ctx_map:
                if not hetero_graph:
                    print('create context:', self._ctx_map)
                    self._ctx_map[func.__name__] = Context(func, nspace, self._run_cb)
                else:
                    raise NotImplementedError('Heterogeneous graph is not supported yet')
            return self._ctx_map[func.__name__]
        return wrapper