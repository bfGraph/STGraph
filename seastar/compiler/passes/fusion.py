from collections import deque
from .cf import CF
from .cse import CSE
from ..utils import ValType, ParallelMode, is_const_scalar, FusionType
from ..program import Program
import copy
from ..execution_unit import ExecutionUnit
from .dependency_analysis import dep_program
from datetime import datetime

class FusionStateMachine():
    state_trans = {
        0 :  { 'a2d' : 1, 'a2s' : 1, 'd' : 3, 's' : 5 , 'e' : 1},
        1 : { 'e' : 1, 's' : 2, 'd' : 2, },
        2 : { 's' : 2, 'd' : 2, },
        3 : { 'd' : 3, 'a2d' : 1, 's' : 4, },
        4 : { 's' : 4, 'd' : 4, },
        5 : { 's' : 5, 'a2s' : 1, 'd' : 4, }
    }
    def __init__(self, init_stmt=None):
        self.cur = 0
        if init_stmt:
            self.accept(init_stmt)
    
    def accept(self, stmt):
        trans = FusionStateMachine.stmt_to_trans(stmt)
        print('cur', self.cur, 'trans', trans, stmt, 'transitionable?', trans in FusionStateMachine.state_trans[self.cur])
        return trans in FusionStateMachine.state_trans[self.cur]

    def advance(self, stmt):
        trans = FusionStateMachine.stmt_to_trans(stmt)
        if trans in FusionStateMachine.state_trans[self.cur]:
            self.cur = FusionStateMachine.state_trans[self.cur][trans]
            return True
        return False
    
    @staticmethod
    def stmt_to_trans(stmt):
        trans = ''
        if stmt.is_agg():
            trans = 'a2d' if stmt.ret.is_dstvar() else 'a2s'
        elif stmt.is_edgewise():
            trans = 'e'
        elif stmt.is_src():
            trans = 's'
        elif stmt.is_dst():
            trans = 'd'
        else:
            raise NotImplementedError('Unknown stmt graph type ' +  stmt)
        return trans
    
    def current_fusion_type(self):
        if self.cur in {3, 4, 5}:
            return FusionType.NN
        elif self.cur in {1, 2}:
            return FusionType.NEAN
        else:
            return FusionType.NOT_FUSIBLE

def mergable(prog1, prog2):
    var_id1 = {var.id for var in prog1.input_vars()}
    var_id2 = {var.id for var in prog2.input_vars()}
    return var_id1.issubset(var_id2) or var_id2.issubset(var_id1)
    #return var_id1 == var_id2

def merge_program(prog_list):
    share_input_map = {}
    for i in range(len(prog_list)):
        share_input_map[i] = set()
        for j in range(i+1, len(prog_list)):
            if mergable(prog_list[i], prog_list[j]):
                share_input_map[i].add(j)

    print('merge_program', share_input_map)
    merged_set = set()
    ret_list = []    
    for i, shared_set in share_input_map.items():
        if i in merged_set:
            continue
        merged_set.add(i)
        ret_list.append(Program())
        ret_list[-1].copy_append_prog(prog_list[i])
        for j in shared_set:
            ret_list[-1].copy_append_prog(prog_list[j])
            merged_set.add(j)
    for prog in ret_list:
        '''Remove redundant computation in fused program'''
        CSE(prog)
    return ret_list

def find_var(var, prog_list):
    for prog in prog_list:
        ret = prog.find_ret_var_by_id(var.id)
        if ret:
            return ret
    return None

def fusable(downstream_s, upstream_s, stmt2state_machine, new_fsm):
    if 'gtypecast' in upstream_s.op_name.lower():
        # type casts are fusion breaker
        return False
    a = downstream_s.is_supported()
    b = upstream_s.is_supported()
    if a and b:
        fsm = stmt2state_machine[downstream_s]
        if fsm.accept(upstream_s):
            if new_fsm:
                nfsm = copy.deepcopy(fsm)
                nfsm.advance(upstream_s)
                stmt2state_machine[upstream_s] = nfsm
            else:
                fsm.advance(upstream_s)
                stmt2state_machine[upstream_s] = fsm
            return True
    elif downstream_s.is_nodewise() and upstream_s.is_nodewise():
        return True
    return False

def merge_stmt(cur_stmt, p_stmt, stmt2fused_prog, stmt2state_machine, stmt_stack, var_stack, prog_list, new_fsm):
    if 'gtypecast' in cur_stmt.op_name.lower():
        stmt_stack.append(p_stmt)
        return
    if fusable(cur_stmt, p_stmt, stmt2state_machine, new_fsm):
        print('fusable', cur_stmt, 'current state:', stmt2state_machine[cur_stmt].cur, 'with p_stmt:', p_stmt, 'new_fsm', new_fsm)
        if p_stmt in stmt2fused_prog:
            if cur_stmt in stmt2fused_prog:
                cur_prog = stmt2fused_prog[cur_stmt]
                if stmt2fused_prog[p_stmt] == cur_prog:
                    print('cur_stmt is already in prog')
                    return
                stmt2fused_prog[p_stmt].copy_append_prog(cur_prog)
                for stmt in cur_prog:
                    stmt2fused_prog[stmt] = stmt2fused_prog[p_stmt]
                cur_prog.clear_stmts()
            else:
                stmt2fused_prog[p_stmt].copy_append_stmt(cur_stmt)
                stmt2fused_prog[cur_stmt] = stmt2fused_prog[p_stmt]
        else:
            if cur_stmt in stmt2fused_prog:
                stmt2fused_prog[cur_stmt].copy_prepend_stmt(p_stmt)
                stmt2fused_prog[p_stmt] = stmt2fused_prog[cur_stmt]
            else:
                var_prog = Program()
                var_prog.copy_append_stmts([p_stmt, cur_stmt])
                stmt2fused_prog[p_stmt] = var_prog
                stmt2fused_prog[cur_stmt] = var_prog
                prog_list.append(var_prog)
            stmt_stack.append(p_stmt)
    else:
        print('not fusable', cur_stmt, 'current state:', stmt2state_machine[cur_stmt].cur, 'with p_stmt:', p_stmt, 'new_fsm', new_fsm)
        if cur_stmt not in stmt2fused_prog:
            var_prog = Program()
            var_prog.copy_append_stmt(cur_stmt)
            stmt2fused_prog[cur_stmt] = var_prog
            prog_list.append(var_prog)
        if p_stmt not in stmt2fused_prog:
            var_stack.append(p_stmt.ret)

def unit_independent(u1, u2):
    return not u1.depends_on(u2) and not u2.depends_on(u1)

def merge_independent(exec_units):
    # Merge units that share the same inputs and has no dependency among each other
    # Check dependency and propose candidate
    candidates = {}
    for i in range(len(exec_units)):
        candidates[i] = []
        for j in range(i+1, len(exec_units)):
            if unit_independent(exec_units[i], exec_units[j]) and exec_units[i].compiled == exec_units[j].compiled:
                candidates[i].append(j)

    # Merge candidates
    merged_units = []
    merged_set = set()
    print('merge independent', candidates)
    for tar_id, src_id_list in candidates.items():
        if tar_id in merged_set:
            continue
        tar_unit = exec_units[tar_id]
        for sid in src_id_list:
            src_unit = exec_units[sid]
            print('src-dst program:', src_unit, tar_unit)
            tar_unit.merge_with_independent_unit(src_unit)
            merged_set.add(sid)
        merged_units.append(tar_unit)
    return merged_units

def fuse(progs, outputs):
    '''
        Generate one/multiple execution units from one or more programs, which are used for code generation.
        Parallel mode of execution unit is determined by the ValType of ret var.
     '''
    if len(progs) == 0:
        return progs
    
    if len(progs) > 1:
        print('-----Program Fusion------')
        progs = merge_program(progs)

    print('-----Operator Fusion------')
    # Starting from each output var, fuse as many operators as possible according to dependenies.
    # Use DFS-manner to allow maximal locality of statements
    stmt2fused_prog = {}
    stmt2state_machine = {}
    prog_list = []
    var_list = []
    for var in outputs:
        ret = find_var(var, progs)
        if ret:
            var_list.append(ret)
    var_list.sort(key=lambda var: var.int_id)
    var_stack = deque(var_list)
    print('sorted var_list', var_list, 'var_stack', var_stack)
    while var_stack:
        var = var_stack.pop()
        stmt_stack = deque([var.stmt])
        while stmt_stack:
            cur_stmt = stmt_stack.pop()
            dep_stmts = []
            for arg in cur_stmt.args:
                if not is_const_scalar(arg) and arg.stmt is not None:
                    dep_stmts.append(arg.stmt)
            if len(dep_stmts) == 0:
                if cur_stmt not in stmt2fused_prog:
                    var_prog = Program()
                    var_prog.copy_append_stmt(cur_stmt)
                    stmt2fused_prog[cur_stmt] = var_prog
                    prog_list.append(var_prog)
                continue
            if cur_stmt not in stmt2state_machine:
                stmt2state_machine[cur_stmt] = FusionStateMachine(cur_stmt)
            print('cur_stmt', cur_stmt, 'current state', stmt2state_machine[cur_stmt].cur, 'dep_sttms', dep_stmts)
            if len(dep_stmts) == 1:
                p_stmt = dep_stmts[0]
                merge_stmt(cur_stmt, p_stmt, stmt2fused_prog, stmt2state_machine, stmt_stack, var_stack, prog_list, new_fsm=False)
            elif len(dep_stmts) == 2:
                l_stmt = dep_stmts[0]
                r_stmt = dep_stmts[1]
                if l_stmt.depends_on(r_stmt):
                    merge_stmt(cur_stmt, l_stmt, stmt2fused_prog, stmt2state_machine, stmt_stack, var_stack, prog_list, new_fsm=False)
                elif r_stmt.depends_on(l_stmt):
                    merge_stmt(cur_stmt, r_stmt, stmt2fused_prog, stmt2state_machine, stmt_stack, var_stack, prog_list, new_fsm=False)
                else:
                    merge_stmt(cur_stmt, r_stmt, stmt2fused_prog, stmt2state_machine, stmt_stack, var_stack, prog_list, new_fsm=True)
                    merge_stmt(cur_stmt, l_stmt, stmt2fused_prog, stmt2state_machine, stmt_stack, var_stack, prog_list, new_fsm=True)
            else:
                raise NotImplementedError('Currenty we assume num of oprands of all operators is no larger than 2')
    prog_l = []
    for p in prog_list:
        prog = Program()
        prog.copy_append_stmts(sorted(p, key=lambda x : x.ret.int_id))
        prog_l.append(prog)
        print(prog)

    prog_blks = [prog for prog in reversed(prog_l) if len(prog) > 0] 

    print('------Construct Execution Unit------')
    exe_units = []
    for prog in prog_blks:
        args = set()
        tmps = set()
        seen_vars = set()
        compiled = False 
        for stmt in prog:
            if not stmt.is_nodewise():
                compiled = True
            for arg in stmt.args:
                if arg not in seen_vars and not is_const_scalar(arg):
                    seen_vars.add(arg)
                    args.add(arg)
            seen_vars.add(stmt.ret)
            tmps.add(stmt.ret)
        exe_units.append(ExecutionUnit(args, tmps, prog, compiled))

    # Connecting units by setting their ret vars
    for i,b in enumerate(exe_units):
        if i > 0:
            for arg in b._args:
                for j in range(i):
                    if arg in exe_units[j].tmps:
                        exe_units[j].add_ret_val(arg)
                        exe_units[i].add_parent_unit(exe_units[j])

    # Set the final outputs for each execution unit
    for unit in exe_units:
        for var in outputs:
            if var in unit.tmps:
                unit.add_ret_val(var)

    # Materialize aggregation results for backward use (s.b.j. to mem-planning) except sum
    for u in exe_units:
        for stmt in u.program:
            if stmt.is_agg():
                if 'sum' not in stmt.op_name.lower():
                    u.add_ret_val(stmt.ret)
                if stmt.ret.is_dstvar():
                    u.set_parallel_mode(ParallelMode.DstParallel)
                elif stmt.ret.is_srcvar():
                    u.set_parallel_mode(ParallelMode.SrcParallel)

    # Sort by return id in order to satisfy the dependency between execution unit
    # Correctness remains quesationable
    exe_units.sort(key=lambda x:x.max_ret_id())
    exe_units = merge_independent(exe_units)
    return exe_units
