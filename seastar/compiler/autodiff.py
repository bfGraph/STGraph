from collections import deque, defaultdict
import copy

from .program import Program, Stmt, Var
from .utils import is_const_scalar
from .passes.dependency_analysis import dep_program
from .passes.cse import CSE
from .passes.cf import CF
from .passes.mem_planning import mem_planning
from .passes import optimize, fuse, visualize

def diff(vars, grads, forward_units, fprog):
    '''
    For each var we find the statment that computes it, then we use itself as well as its grad
    to get the statements to calculate or accumulate the gradient for each of its imputs. 

    We need to synchrounize the graidents for a same var in order to keep the statements ordered 
    by return var id. In order to do that, we use processed_count == forward_num_users_of_var as 
    condition to determine whether we can push it to queue. If processed_count < forward* it means
    that we need to wait for computing more of its gradients before propogate back through that 
    variable. If the var is in stopping var, there is no need to propogate further as the task will
    be delegated to backend system.
    
    args:
        vars : the var that has gradient to propogate back. Determined by zoomOut
        grads : the coresponding gradient for each var
        forward_units : forward execution units

    return:
        BProg : differentiated program to compute the gradients
        grad_map : the gradient map of vars
    '''
    assert len(vars) == len(grads), 'Each var must have a corresponding grad'
    BProg = Program()
    q = deque() 
    grad_map = {}
    var_set = set()
    for i, var in enumerate(vars):
        q.append(var)
        grad_map[var.id] = grads[i]
        var.set_grad(grads[i])
        var_set.add(var)
    processed_count = defaultdict(int)
    forward_num_users_of_var = {}
    forward_args = set()
    stopping_var = set()

    for unit in forward_units:
        forward_args = forward_args.union(unit.get_all_args())
    for var in forward_args:
        forward_num_users_of_var[var.id] = len(var.users)
    for unit in forward_units:
        if unit.compiled == False:
            for stmt in unit.program:
                for var in stmt.args:
                    if not is_const_scalar(var):
                        processed_count[var] += sum([1 if unit.program.has_stmt(stmt) else 0 for stmt in var.users])
            stopping_var = stopping_var.union(set([ret.id for ret in unit.all_rets()]))
    print('\n------------Autodiff: Retrive BackwardProg-----------\n')
    while q:
        y = q.pop()
        cur_stmt = y.stmt
        # Get the statements for each input x
        if y.id in stopping_var or cur_stmt == None:
            if cur_stmt:
                for arg in cur_stmt.args:
                    if not is_const_scalar(arg) and arg.requires_grad:
                        print('Handled by DL backend. Skipping to next var:', arg)
                        q.append(arg)
            continue
        if 'gtypecast' in cur_stmt.op_name.lower():
            # Skip gtypecast
            x = cur_stmt.args[0]
            grad_map[y.id]._val_type = x.val_type
            print('differentiating gtypecast:', cur_stmt, ' using y:', y, 'grad_y:', grad_map[y.id])
            q.append(cur_stmt.args[0])
            continue
        #if y not in grad_map:
        #    # The unusual case, where two compiled kernels are seperated
        #    grad_map[y] = Var.create_var(var_shape=y.var_shape, var_dtype=y.var_dtype, val_type=y.val_type, device=y.device, requires_grad=y.requires_grad)
        #    print('Create grad', grad_map[y], 'for', y, ',who is produced by', y.stmt)
        grad_y = grad_map[y.id]
        x2stmts = cur_stmt.grad(y, grad_y)
        print('\ndifferentiating:', cur_stmt, ' using y:', y, 'grad_y:', grad_y)
        x2stmt_list = list(x2stmts.items())
        for x, stmts in x2stmt_list:
            # Compute gradient
            for stmt in stmts:
                BProg.append_stmt(stmt)
                var_set.add(stmt.ret)
            # Accumulate/Record gradient for inputs
            if x.id in grad_map:
                acc_stmt = Stmt.create_add_stmt([grad_map[x.id], stmts[-1].ret])
                BProg.append_stmt(acc_stmt)
                stmts.append(acc_stmt)
                grad_map[x.id] = acc_stmt.ret
                x.set_grad(acc_stmt.ret)
                var_set.add(acc_stmt.ret)
            else:
                grad_map[x.id] = stmts[-1].ret
                x.set_grad(stmts[-1].ret)
            # Propagate back further. Each var is propogated only once
            processed_count[x] += 1
            if processed_count[x] == forward_num_users_of_var[x.id]:
                q.append(x)

    need_grad_var = set()
    output_var = set()
    for unit in forward_units:
        if unit.compiled:
            for arg in unit.unit_args():
                if arg not in output_var and arg.requires_grad:
                    need_grad_var.add(fprog.find_var_by_id(arg.id))
                    arg._grad = fprog.find_var_by_id(arg.id)._grad
            for ret in unit.unit_rets():
                output_var.add(ret)
    print('\n------------Autodiff: Optimizing BackwardProg-----------\n')
    optimize(BProg)
    #visualize.plot_exec_units(forward_units)
    #visualize.plot_programs([unit._prog for unit in forward_units] + [BProg])
    print('\n------------Autodiff: Gradient Driven MemPlanning-----------\n')
    output_grad_map = {k:k._grad for k in need_grad_var}
    bp_prog_list = mem_planning(forward_units, BProg, output_grad_map, grads)
    print('\n------------Autodiff: Optimizing Programs of Each Gradient-----------\n')
    for prog in bp_prog_list:
        optimize(prog)
    print('\n------------Autodiff: Fusing Programs of Each Gradient-----------\n')
    backward_exe_units = fuse(bp_prog_list, [v for _, v in output_grad_map.items()])
    print('\n------------Autodiff: Done-----------\n')
    return backward_exe_units
