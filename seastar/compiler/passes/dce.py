def DCE(prog, output_vars):
    '''
        DCE, modify prog in place.
        A statement is marked dead if there is no use of its return value
        and its return value is not used as outputs
    '''
    for s in reversed(prog): 
        if len(s.ret.users) == 0 and s.ret not in output_vars:
            s.remove_cur()
    print('After DCE programs becomes:', prog)