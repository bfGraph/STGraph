import torch
from collections import defaultdict, namedtuple
from collections.abc import Iterable
import sys

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

class SkipContextException(Exception):
    pass

class zoomIn(object):
    def __init__(self, namespace, input_map, node_feats, edge_feats, fprog):
        self._namespace = namespace
        self._backend = self._find_backend(namespace)
        self._nfeats = node_feats 
        self._efeats = edge_feats
        self._v = CentralNode()
        self._old_mapping = defaultdict(dict)
        self.inputs = input_map
        self.fprog = fprog
        self.entry_count = 0
    
    def clear_cache(self):
        if self.inputs:
            del self._namespace
            del self._backend
            del self._nfeats
            del self._efeats
            del self._v
            del self._old_mapping
            del self.inputs
            del self.fprog

    def __enter__(self):
        if self.entry_count == 0:
            print('entering local context')
            self._symbolization()
            self.entry_count += 1
            return self._v
        else:
            sys.settrace(lambda *args, **kargs: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace
        
    def trace(self, frame, event, arg):
        raise SkipContextException

    def __exit__(self, ctx_type, ctx_value, ctx_traceback):
        if ctx_type == SkipContextException:
            return True
        self._desymbolization()
        print('TracedProgram:\n' + str(self.fprog))
    
    def mapping_key(self, name_space_id, original_key):
        return str(name_space_id) + str(original_key)

    def _symbolization(self):
        """Symbolizing central node and its innbs and inedges""" 
        var_name = ''
        for k, v in self._nfeats.items():
            setattr(self._v, k, create_dst_node_val(v, self._backend, id=k+cen_attr_postfix, fprog=self.fprog))
            var_name += '%' + k  + ','
            for n in self._v.innbs:
                setattr(n, k, create_src_node_val(v, self._backend, id=k+inb_attr_postfix, fprog=self.fprog))
        for k, v in self._efeats.items():
            var_name += '%' + k  + ','
            for e in self._v.inedges:
                setattr(e, k, create_edge_val(v, self._backend, id=k, fprog=self.fprog))
        self.fprog.set_input(var_name.strip(','))
        if self._backend[0] == 'torch':
            for i, nspace in enumerate(self._namespace):
                if '__name__' in nspace.__dict__:
                    # symbolizing functions for torch namespace
                    k = self.mapping_key(i, 'function')
                    for key in nspace.__dict__:
                        m = nspace.__dict__[key]
                        if 'function' in str(type(m)):
                            if key in self._old_mapping[k]:
                                raise KeyError('Found', key, ' already in _old_mapping')
                            self._old_mapping[k][key] = m
                            nspace.__dict__[key] = create_op(m, self._backend[0], fprog=self.fprog)
                else:
                    ## Dealing with module
                    for key in nspace.__dict__.keys():
                        # symbolizing parameters for self namespace
                        m = nspace.__dict__[key]
                        k = self.mapping_key(i, key)
                        if key.startswith('_parameters'):
                            for mkey in m.keys():
                                if mkey in self._old_mapping[k]:
                                    raise KeyError('Found', key, ' already in _old_mapping')
                                self._old_mapping[k][mkey] = m[mkey] 
                                self.inputs[var_prefix+mkey] = m[mkey]
                                m[mkey] = create_param_val(m[mkey], self._backend, id=mkey, fprog=self.fprog)

                        # symbolizing buffers for self namespace
                        if key.startswith('_buffers'):
                            for mkey in m.keys():
                                if mkey in self._old_mapping[k]:
                                    raise KeyError('Found', key, 'already in _old_mapping')
                                self._old_mapping[k][mkey] = m[mkey]
                                self.inputs[var_prefix+mkey] = m[mkey]
                                m[mkey] = create_param_val(m[mkey], self._backend, id=mkey, fprog=self.fprog)

                        # symbolizing modules for self namespace
                        if key.startswith('_modules'):
                            for mkey in m.keys():
                                if mkey in self._old_mapping[k]:
                                    raise KeyError('Found', key, ' already in _old_mapping')
                                self._old_mapping[k][mkey] = m[mkey]
                                m[mkey] = create_op(m[mkey], self._backend[0], fprog=self.fprog)
        else:
            raise NotImplementedError('Backend ' + self._backend[0] + ' is not supported yet!') 

    def _desymbolization(self):
        if self._backend[0] == 'torch':
            for i, nspace in enumerate(self._namespace):
                if '__name__' in nspace.__dict__:
                    # desymbolizing functions for torch namespace
                    if 'torch' in nspace.__name__.lower():
                        k = self.mapping_key(i, 'function')
                        for key in self._old_mapping[k]:
                            nspace.__dict__[key] = self._old_mapping[k][key]
                else:
                    for key in nspace.__dict__.keys():
                        # desymbolizing parameters for self namespace
                        m = nspace.__dict__[key]
                        k = self.mapping_key(i, key) 
                        if key.startswith('_parameters') or key.startswith('_modules') or key.startswith('_buffers'):
                            for mkey in self._old_mapping[k]:
                                m[mkey] = self._old_mapping[k][mkey]

        else:
            raise NotImplementedError('Backend ' + self._backend[0] + ' is not supported yet!') 

    def _find_backend(self, namespace):
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
    
    def outputs(self):
        return self._v.outputs

def zoomOut(out_set, fprog, graph):
    optimize(fprog)
    vars = []
    for var in out_set:
        vars.append(var)
    forward_exe_units = fuse([fprog], vars)
    grads = []
    for var in vars:
        grads.append(Var.create_var(var_shape=var.var_shape, var_dtype=var.var_dtype, val_type=var.val_type, device=var.device))
    backward_exe_units, full_grad_map = diff(vars, grads, forward_exe_units)
    visualize.plot_exec_units(forward_exe_units + backward_exe_units)
    compiled_module = code_gen.gen_code(forward_exe_units + backward_exe_units, 'int' if graph.nbits == 32 else 'long long int')
    ex = Executor(graph, forward_exe_units, backward_exe_units, compiled_module, full_grad_map, vars)
    return ex 

class ContextManager(object):
    GraphInfo = namedtuple('GraphInfo', ['number_of_nodes',
                                         'number_of_edges',
                                         'in_row_offsets',
                                         'in_col_indices',
                                         'in_eids',
                                         'out_row_offsets',
                                         'out_col_indices',
                                         'out_eids',
                                         'nbits'])
    graph_info = None
    cached_graph = None
    reset = False

    def __init__(self, run_egl):
        self._executor = None
        self._ctx = None
        self.input_map = {}
        self._fprog = Program()
        self.run_egl_cb = run_egl
        self.outputs = set()

    def zoomIn(self, namespace=None, graph=None, node_feats={}, edge_feats={}):
        if ContextManager.graph_info == None or id(graph) != id(ContextManager.cached_graph):
            ContextManager.cached_graph = graph
            ContextManager.reset = True
            ## Assuming different context are working on the same graph
            # This may not be true for mini-batch training
            in_csr = graph.get_in_csr()
            out_csr = graph.get_out_csr()
            ContextManager.graph_info = ContextManager.GraphInfo(
                graph.number_of_nodes(),
                graph.number_of_edges(),
                in_csr(0).copy_to_gpu(0),
                in_csr(1).copy_to_gpu(0),
                in_csr(2).copy_to_gpu(0),
                out_csr(0).copy_to_gpu(0),
                out_csr(1).copy_to_gpu(0),
                out_csr(2).copy_to_gpu(0),
                graph.nbits()
            )
        for k, v in node_feats.items():
            self.input_map[var_prefix + k + cen_attr_postfix] = v
            self.input_map[var_prefix + k + inb_attr_postfix] = v
        for k, v in edge_feats.items():
            self.input_map[var_prefix+k] = v
        if not self._ctx:
            self._ctx = zoomIn(namespace, self.input_map, node_feats, edge_feats, self._fprog)
        return self._ctx

    def zoomOut(self):
        if not self._executor:
            self._executor = zoomOut(self.outputs, self._fprog, ContextManager.graph_info)
            ContextManager.reset = False
        self._executor.restart(self.input_map, ContextManager.graph_info if ContextManager.reset else None)
        ret = self.run_egl_cb(self._executor)
        if len(ret) == 1:
            return ret[0]
        self.input_map.clear()
        self._ctx.clear_cache()
        return ret

    def collect_output(self, val):
        if isinstance(val, list):
            for v in val:
                self.outputs.add(v.var)
        else:
            self.outputs.add(val.var)
    
    def update_allnode(self, dict):
        cen=self._ctx._v
        for k,v in dict.items():
            if k in cen.__dict__:
                cen.__dict__[k] = v
            for nb in cen.innbs:
                if k in nb.__dict__:
                    # Here we re-use the var id of central node to indicate the fact that
                    # they are the same tensor with different access modifier(src and dst)
                    nb.__dict__[k] = create_src_node_val(v._v, v.backend, id=str(v.var.int_id), fprog=self._fprog, reduce_dim=False)
                    nb.__dict__[k].var._stmt = Stmt.create_stmt(op_schema=Schema('GTypeCast'), args=[v.var], ret=nb.__dict__[k].var) 
                    self._fprog.append_stmt(nb.__dict__[k].var._stmt)
                    nb.__dict__[k].var._requires_grad = v.var._requires_grad

    def max(self, args):
        assert isinstance(args, list), 'Can only aggregate feature list'
        max_op = AggMaxOp()
        return max_op(self._fprog, args)

    def min(self, args):
        assert isinstance(args, list), 'Can only aggregate feature list'
        min_op = AggMinOp()
        return min_op(self._fprog, args)

    def mean(self, args):
        assert isinstance(args, list), 'Can only aggregate feature list'
        mean_op = AggMeanOp()
        return mean_op(self._fprog, args)
