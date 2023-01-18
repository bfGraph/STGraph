from collections import deque
from ..program import Program, Stmt, Var
from ..utils import is_const_scalar

def dep_program(target_var, stopping_vars):
    print('dep_program', stopping_vars)
    '''
        Step1: Analyze the part of the program that computes a "target_var".
        Recursively 1. find the stmt that creates the current var 2. find the
        dependant stmts of its inputs until encounter "stopping_vars".

        Step2: Set the dependent stmt of stopping var to be None

        Returns a copy of the dependency program of target_var 
    '''
    q = deque()
    q.append(target_var)
    dep = []
    seen_stmt = set()
    while q:
        targ_var = q.pop()
        stmt = targ_var.stmt
        if stmt and (stmt,targ_var) not in seen_stmt:
            seen_stmt.add((stmt, targ_var))
            dep.append(stmt)
            for var in stmt.args:
                if var not in stopping_vars and not is_const_scalar(var):
                    q.append(var)

    prog = Program()
    prog.copy_append_stmts(sorted(reversed(dep), key=lambda x : x.ret.int_id))

    for stmt in prog:
        for var in stmt.args:
            if var in stopping_vars:
                print('setting ', var, 'stmt to be None')
                var.stmt = None # Stopping further propogation
    return prog