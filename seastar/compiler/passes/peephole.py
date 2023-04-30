from sympy import *
import re
from collections import deque
from ..utils import is_const_scalar, bcast_dim, infer_val_type, var_prefix
from .dependency_analysis import dep_program
from .cse import CSE
from .dce import DCE
from ..program import Stmt, Var
from ..schema import Schema

def execute_sym_program(prog, sym_table, rmv_list):
    for s in prog:
        if 'mul' in s.op_name.lower():
            l = sym_table[s.args[0].id] if not is_const_scalar(s.args[0]) else s.args[0] 
            r = sym_table[s.args[1].id] if not is_const_scalar(s.args[1]) else s.args[1] 
            sym_table[s.ret.id] = l * r  
        elif 'div'in s.op_name.lower():
            l = sym_table[s.args[0].id] if not is_const_scalar(s.args[0]) else s.args[0] 
            r = sym_table[s.args[1].id] if not is_const_scalar(s.args[1]) else s.args[1] 
            sym_table[s.ret.id] = l / r
        elif 'sum' in s.op_name.lower():
            # sum and aggsum are fusion breaker.
            # We ensure that before and after sum or aggsum is a series of mul/div,
            # we leverage the distributive and associative nature of sum and mul, to ignore the sum operator
            if 'agg' not in s.op_name.lower():
                rmv_list.append(s)
            l = sym_table[s.args[0].id] if not is_const_scalar(s.args[0]) else s.args[0] 
            sym_table[s.ret.id] = l
        else:
            print('early stopping due to encounter', s)
            break

def generate_stmts_from_expr(expr, var_table):
    print('var_table', var_table)

    preceding_neg = False
    if expr[0] == '-':
        preceding_neg = True
        expr = expr[1:]
    tok_list = re.split('([-*/])', expr)
    tok_q = deque(tok_list)
    stmt_list = []
    while len(tok_q) > 1:
        arg0 = var_table[tok_q.popleft()]
        op = tok_q.popleft()
        arg1 = var_table[tok_q.popleft()]
        print('to create stmt for', arg0, op, arg1)
        if op == '*':
            op_name = 'Mul'
        elif op == '/':
            op_name = 'TrueDiv'
        else:
            raise NotImplementedError('op', op, 'is not supprted for PH optimization')
        stmt_list.append(Stmt.create_binary_bcast_stmt(Schema(op_name), args=[arg0, arg1]))
        print('after create binary_bcast stmt', arg1)
        ret = stmt_list[-1].ret
        var_table[ret.id] = ret
        tok_q.appendleft(ret.id)
    if preceding_neg:
        stmt_list.append(Stmt.create_binary_bcast_stmt(Schema('Mul'), args=[-1, stmt_list[-1].ret]))
    return stmt_list

def shape_propogation(s):
    if 'agg' not in s.op_name.lower():
        # Effectively, we merge the sum op with aggsum op
        dim = bcast_dim(s.args)
        if dim != s.ret.var_shape:
            s.ret.var_shape = dim
            for stmt in s.ret.users:
                shape_propogation(stmt)

def sum_propogation(sum_stmt):
    ret = sum_stmt.ret
    arg = sum_stmt.args[0]
    for stmt in ret.users:
        for i in range(len(stmt.args)):
            old_arg = stmt.args[i]
            if old_arg == ret:
                # Replace sum ret with sum arg
                print(old_arg, 'is replaced with', arg, 'in stmt', stmt)
                stmt.args[i] = arg
                print('after replacement', stmt)
                shape_propogation(stmt)

def PH(BProg, known_vars, output_vars):
    ''' 
        Peephole optimization pass. Mofiy program in place.
        It replaces unfusable expressions('sum' and 'aggsum') in forward and backward programs with known_vars
        by applying various mathmatically equivelent tansformations.
    '''
    # Find candidate chain breaker;
    print('Before PH', BProg)
    candidate_vars = set()
    var_table = {}
    for s in BProg:
        if 'sum' in s.op_name.lower() or 'agg' in s.op_name.lower():
            if s.ret not in output_vars:
                candidate_vars.add(s.ret)
                var_table[s.ret.id] = s.ret
    sym_table = {}
    for var in known_vars:
        sym_table[var.id] = Symbol(var.id) 
        var_table[var.id] = var

    sum_map = {}
    for var in known_vars:
        dep_prog = dep_program(var, known_vars - set([var]))
        sum_map[var.id] = []
        execute_sym_program(dep_prog, sym_table, sum_map[var.id])

    for var in candidate_vars:
        dep_prog = dep_program(var, known_vars)
        sum_map[var.id] = []
        execute_sym_program(dep_prog, sym_table, sum_map[var.id])
    print('candidate_vars', candidate_vars)

    simplifiable_expression = []
    for cv in candidate_vars: 
        for kv in known_vars:
            if cv.id in sym_table and kv.id in sym_table:
                var1 = sym_table[cv.id]
                var2 = sym_table[kv.id]
                var3 = var1/var2
                if 'mul' in str(type(var3)).lower() and str(var3).count(var_prefix) < str(var1).count(var_prefix) - 1:
                    print(var1, '/', var2, '=', var3)
                    simplifiable_expression.append((cv.id, str(var3) +'*'+kv.id, sum_map[cv.id]))
    print('simplifiable expr:', simplifiable_expression)
    for tu in simplifiable_expression:
        target_var = var_table[tu[0]]
        stmts = generate_stmts_from_expr(tu[1], var_table)
        BProg.insert_stmts_before(target_var.stmt, stmts)
        # Replace last op
        target_var.replace_all_uses_with(stmts[-1].ret, propogate_shape=False)
        print('new prog after inserting generated stmts', BProg)
        DCE(BProg, output_vars)
        for s in tu[2]:
            for st in BProg:
                if st == s:
                    print('trying to remove', st)
                    # Omit the last stmt as it's already be removed
                    print('s.ret.users', st.ret.users)
                    sum_propogation(st)
                    st.remove_cur()
        BProg.resort_vars()
        print('Final bprog:', BProg)
