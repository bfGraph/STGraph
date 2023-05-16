from .dependency_analysis import dep_program
from .peephole import PH
from ..program import Program
from ..utils import is_const_scalar

from seastar.compiler.debugging.seastar_logger import print_log

def mem_planning(funits, BProg, grad_map, grads):
    ''' 
        Conduct memory planning by considering both FProg and BProg
        Create annotations for the graph so that code-generation can be done
        TODO:
        Add cost-model to choose a different materialized vars
    '''

    materialized_vars = set(grads)
    for unit in funits:
        materialized_vars = materialized_vars.union(unit.materilized_vars())

    print_log("[red bold]Peephole[/red bold]: Starting Peephole optimization")

    bp_list = []
    for var, grad in grad_map.items():
        bp_list.append(dep_program(grad, stopping_vars=materialized_vars))

    for bp in bp_list:
        PH(bp, materialized_vars, set([grad_map[key] for key in grad_map]))
        
    print_log("[red bold]Peephole[/red bold]: Peephole optimization completed")
    return bp_list