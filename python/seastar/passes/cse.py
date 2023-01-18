def CSE(prog):
    '''
        Common subexpression elimination
        Modify prog in place
    '''
    s_map = {}
    for stmt in prog:
        info = stmt.stmt_info()
        if info in s_map:
            ret_var = s_map[info]
            if stmt.ret:
                print("CSE: replacing" ,stmt.ret, 'with', ret_var, 'in', info, stmt)
                stmt.ret.replace_all_uses_with(ret_var, propogate_shape=False)
            stmt.remove_cur()
        else:
            s_map[info] = stmt.ret
    print('After apply CSE', prog)