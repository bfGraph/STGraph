from .template import gen_cuda, EdgeInfo, AggInfo, ArgInfo, NodeInfo
from ..utils import is_const_scalar, ParallelMode

const_id = 0
def gen_arg_info(arg):
    if is_const_scalar(arg):
        global const_id
        arg_info = ArgInfo(name='c'+str(const_id), type=str(type(arg)), is_ptr=False)
        const_id += 1
    else:
        arg_info = ArgInfo(name=arg.id, type=str(arg.dtype_str), is_ptr=True)
    return arg_info

def gen_agg_info(stmt, ctx):
    m = stmt.gen_code(ctx)
    if not m:
        raise NotImplementedError('Cannot generate code for', stmt)
    return AggInfo(**m)

def gen_edge_info(stmt, ctx):
    m = stmt.gen_code(ctx)
    if not m:
        raise NotImplementedError('Cannot generate code for', stmt)
    return EdgeInfo(**m)

def gen_node_info(stmt, ctx):
    m = stmt.gen_code(ctx)
    if not m:
        raise NotImplementedError('Cannot generate code for', stmt)
    return NodeInfo(**m)

def gen_code(exe_units, index_type):
    '''Generating cuda code by instantiate code template'''
    if not isinstance(exe_units, list):
        exe_units = [exe_units]
    configs = []
    for unit in exe_units:
        if not unit.compiled:
            continue
        arginfos = []
        nodeinfos = []
        agginfos = []
        edgeinfos = []
        for var in unit.kernel_args():
            arginfos.append(gen_arg_info(var))
        ctx = unit.create_context(index_type)
        after_agg = False
        for stmt in unit.program:
            ctx.set_stmt_ctx(stmt)
            if stmt.is_agg():
                agginfos.append(gen_agg_info(stmt, ctx))
                after_agg = True
            elif stmt.is_edgewise():
                edgeinfos.append(gen_edge_info(stmt, ctx))
            elif stmt.is_nodewise():
                if after_agg:
                    nodeinfos.append(gen_node_info(stmt, ctx))
                else:
                    edgeinfos.append(gen_edge_info(stmt, ctx))

        dst_parallel = True if unit.parallel_mode() == ParallelMode.DstParallel else False
        configs.append({
            'kernel_name': unit.kernel_name,
            'index_type' : index_type,
            'args': arginfos,
            'edges': edgeinfos,
            'aggs': agginfos,
            'nodes': nodeinfos,
            'row_offset': 'dst_id' if dst_parallel else 'src_id',
            'init_outter_offset': ctx.param_offset_init + (ctx.dst_var_offset_init if dst_parallel else ctx.src_var_offset_init),
            'col_index': 'src_id' if dst_parallel else 'dst_id',
            'init_inner_offset': (ctx.src_var_offset_init if dst_parallel else ctx.dst_var_offset_init) + ctx.edge_var_offset_init,
            'template_name': ctx.template_name,
        })
    return gen_cuda(configs)
