import networkx as nx
from ..utils import is_const_scalar, OpType

count = 0
var_shape='plain'
stmt_shape='box'
color_map= {
    OpType.S : 'lightgreen',
    OpType.E : 'lightblue',
    OpType.D : 'lightyellow',
    OpType.A : 'red',
}

def plog_program(prog):
    plog_programs([prog])

def plot_programs(progs, filename='egl-dag'):
    global count
    G = nx.DiGraph()
    edges = []
    stmt_count = {}
    G.add_node('src op', shape=stmt_shape, color='lightgreen', style='filled')
    G.add_node('edge op', shape=stmt_shape, color='lightblue', style='filled')
    G.add_node('dst op', shape=stmt_shape, color='lightyellow', style='filled')
    G.add_node('agg op', shape=stmt_shape, color='red', style='filled')
    for prog in progs:
        for stmt in prog:
            arg_nodes = [arg if is_const_scalar(arg) else arg.id for arg in stmt.args]
            stmt_node = str(stmt.op_name)
            if stmt_node not in stmt_count:
                stmt_count[stmt_node] = 0
            else:
                stmt_count[stmt_node] += 1
            stmt_node = str(stmt.op_name) + '-' + str(stmt_count[stmt_node])
            ret_node = stmt.ret.id
            # Add nodes
            for arg in arg_nodes:
                G.add_node(arg, shape=var_shape)
            G.add_node(stmt_node, shape=stmt_shape, color=color_map[stmt.op_type], style='filled')
            G.add_node(ret_node, shape=var_shape)
            # Add edges
            for arg in arg_nodes:
                G.add_edge(arg, stmt_node)
            G.add_edge(stmt_node, ret_node)
    p=nx.drawing.nx_pydot.to_pydot(G)
    p.write_svg(filename + str(count) + '.svg')
    count += 1

compiled_color_map = {
    True: 'green',
    False: 'blue'
}

def plot_exec_units(units, filename='egl-fused-dag'):
    global count
    G = nx.DiGraph()
    stmt_set = set()
    G.add_node('fused-and-compiled', shape=stmt_shape, color='green', style='filled')
    G.add_node('not compiled', shape=stmt_shape, color='blue', style='filled')
    for i, unit in enumerate(units):
        arg_nodes = [arg if is_const_scalar(arg) else arg.id for arg in unit.unit_args()]
        unit_node = unit.kernel_name + '\n' + '\n'.join([stmt.op_name for stmt in unit._prog])
        ret_nodes = [ret.id for ret in unit.unit_rets()]
        # Add nodes
        for arg in arg_nodes:
            G.add_node(arg, shape=var_shape)
        G.add_node(unit_node, shape=stmt_shape, color=compiled_color_map[unit.compiled], style='filled')
        for ret in ret_nodes:
            G.add_node(ret, shape=var_shape)
        # Add edges
        for arg in arg_nodes:
            G.add_edge(arg, unit_node)
        for ret in ret_nodes:
            G.add_edge(unit_node, ret)
    p=nx.drawing.nx_pydot.to_pydot(G)
    p.write_svg(filename + str(count) + '.svg')
    count += 1