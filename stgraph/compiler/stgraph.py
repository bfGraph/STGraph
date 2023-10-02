import functools
from collections import defaultdict, namedtuple
from collections.abc import Iterable

from .node import CentralNode
# from .op.op import create_op, AggMaxOp, AggMinOp, AggMeanOp
from .program import Var, Stmt, Program
from .passes import optimize, CF, fuse, visualize
from .schema import Schema
from .autodiff import diff
from .code_gen import code_gen 
from .executor import Executor
from .utils import var_prefix, cen_attr_postfix, inb_attr_postfix
import gc
import torch

from stgraph.compiler.backend.callback import STGraphBackend
from stgraph.compiler.utils import ValType
from stgraph.compiler.val.val_factory import ValFactory
from stgraph.compiler.op.op_factory import OpFactory

import snoop


class Context():
    def __init__(self, func, nspace, run_cb):
        functools.update_wrapper(self, func)
        self._f = func
        self._nspace = nspace
        self._entry_count = 0
        self._run_cb = run_cb
        self.val_factory = ValFactory()
        self._op_factory = OpFactory()
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
        if self._entry_count == 0:
            fprog = Program()
            ret = self._trace(node_feats, edge_feats, self._input_cache, fprog)
            # print('TracedProgram' + str(fprog), 'Ret value:', ret)
            # pretty_print_GIR(fprog,"TGCN GIR")
            self._executor_cache = self._diff_then_compile(ret, fprog, graph)
        
        for k, v in node_feats.items():
            self._input_cache[var_prefix + k + cen_attr_postfix] = v
            self._input_cache[var_prefix + k + inb_attr_postfix] = v
        for k, v in edge_feats.items():
            self._input_cache[var_prefix+k] = v
        self._executor_cache.restart(self._input_cache, graph)
        self._entry_count += 1
        return self._executor_cache

    def _trace(self, nfeats, efeats, input_cache, fprog):
        backend = self._find_backend()
        central_node = self._init_central_node(nfeats, efeats, fprog, backend)
        # pretty_print_Central_Node(central_node=central_node, print_tensors=False)
        old_libs = defaultdict(dict)
        self._monkey_patch_namespace(old_libs, input_cache, fprog, backend)
        ret = self._f(central_node)
        self._remove_patch(old_libs, backend)
        self._destroy_central_node(central_node, nfeats, efeats)

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
        # visualize.plot_exec_units(forward_exe_units + backward_exe_units)
        
        # NOTE: The last parameter here was ('int' if graph.nbits == 32 else 'long long int') but we changed
        # it to just 'int' since that should be sufficient for all use case that we can think of now
        compiled_module = code_gen.gen_code(forward_exe_units + backward_exe_units, 'int', graph.graph_type())
        return Executor(graph, forward_exe_units, backward_exe_units, compiled_module, vars)
        
    def _init_central_node(self, nfeats, efeats, fprog, backend):
        cen = CentralNode()
        if nfeats:
            for k, v in nfeats.items():
                dst_node_val = self.val_factory.create(ValType.DEST, v, backend, id=k+cen_attr_postfix, fprog=fprog, reduce_dim=True)
                setattr(cen, k, dst_node_val)
                for n in cen.innbs:
                    src_node_val = self.val_factory.create(ValType.SRC, v, backend, id=k+inb_attr_postfix, fprog=fprog, reduce_dim=True)
                    setattr(n, k, src_node_val)
        if efeats:
            for k, v in efeats.items():
                for e in cen.inedges:
                    edge_val = self.val_factory.create(ValType.EDGE, v, backend, id=k, fprog=fprog, reduce_dim=True)
                    setattr(e, k, edge_val)
        return cen

    def _destroy_central_node(self, cen, nfeats, efeats):
        if nfeats:
            for k, _ in nfeats.items():
                delattr(cen, k)
                for n in cen.innbs:
                    delattr(n, k)
        if efeats:
            for k, _ in efeats.items():
                for e in cen.inedges:
                    delattr(e, k)
    
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
                            nspace.__dict__[key] = self._op_factory.create(m, backend[0], fprog)
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
                                param_val = self.val_factory.create(ValType.PARAM, m[mkey], backend, id=mkey, fprog=fprog, reduce_dim=False)
                                m[mkey] = param_val

                        # symbolizing buffers for self namespace
                        if key.startswith('_buffers'):
                            for mkey in m.keys():
                                if mkey in old_libs[k]:
                                    raise KeyError('Found', key, 'already in old_libs')
                                old_libs[k][mkey] = m[mkey]
                                input_cache[var_prefix+mkey] = m[mkey]
                                param_val = self.val_factory.create(ValType.PARAM, m[mkey], backend, id=mkey, fprog=fprog, reduce_dim=False)
                                m[mkey] = param_val

                        # symbolizing modules for self namespace
                        if key.startswith('_modules'):
                            for mkey in m.keys():
                                if mkey in old_libs[k]:
                                    raise KeyError('Found', key, ' already in old_libs')
                                old_libs[k][mkey] = m[mkey]
                                m[mkey] = self._op_factory.create(m[mkey], backend[0], fprog)
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
    
    def _find_backend(self):
        """ Finds the backend framework being used
        
            Returns:    A tuple containing the name and module instance of 
                        the backend being used
        """
        backend_module = self._nspace[1]
        backend_name = backend_module.__name__
        return (backend_name, backend_module)

    def _mapping_key(self, name_space_id, original_key):
        return str(name_space_id) + str(original_key)


class STGraph():
    def __init__(self, backend_framework: STGraphBackend):
        self._ctx_map = {}
        self._backend_framework = backend_framework
        self._run_cb = backend_framework.backend_cb
    
    def compile(self, gnn_module, hetero_graph=False):
        
        # adding the GNN module and the backend framework to the namespace list
        namespace = [gnn_module, self._backend_framework.backend_module]
        
        def wrapper(func):
            if not func.__name__ in self._ctx_map:
                if not hetero_graph:
                    self._ctx_map[func.__name__] = Context(func, namespace, self._run_cb)
                else:
                    raise NotImplementedError('Heterogeneous graph is not supported yet')
            return self._ctx_map[func.__name__]
        return wrapper