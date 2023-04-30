def CF(prog):
    ''' Constant folding. Modify prog in place.
        Currently, we replace var2 = mul(var1, 1) with var1
                   we replace the var3 in  var2(valType.NodeType) = aggsum(var1(valType.E))
                                           var3(valType.E) = mul(var2(valType.NodeType), 1)
                   with var1 to bypass the aggsum and mul
    '''
    for cur_stmt in prog:
        # We cannot change the order as mul is a subpattern of aggsum.
        #if cur_stmt.op_name.lower() == 'aggsum':
        #    assert len(cur_stmt.args) == 1
        #    var1 = cur_stmt.args[0]
        #    user_list = list(cur_stmt.ret.users)
        #    for stmt in user_list:
        #        if stmt.op_name.lower() == 'mul' and 1 in stmt.args and stmt.ret.val_type==ValType.E:
        #            stmt.ret.replace_all_uses_with(var1)
        #            stmt.remove_cur()
                    #if len(cur_stmt.ret.users) == 0:
                    #    print('aggsum remove stmt', cur_stmt)
                    #    cur_stmt.remove_cur()

        if cur_stmt.op_name.lower() == 'mul':
            assert len(cur_stmt.args) == 2, 'mul stmt has exactly 2 arguments'
            has_one = -1
            for i,arg in enumerate(cur_stmt.args):
                if arg == 1:
                    has_one = i
            if has_one != -1:
                print('CF: remove', cur_stmt)
                cur_stmt.ret.replace_all_uses_with(cur_stmt.args[1-i], propogate_shape=False)
                cur_stmt.remove_cur()
    print('after cf program becomes:', prog)