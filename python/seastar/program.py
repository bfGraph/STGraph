from collections.abc import Iterable
from collections import namedtuple
from .schema import Schema 
from .utils import infer_op_type_from_args, val_seq, infer_val_type, is_const_scalar, OpType, ValType, bcast_dim, any_var, unused_ids, var_prefix, inb_attr_postfix, cen_attr_postfix
from .registry import look_up_registry, register_or_look_up_backend_cb

class Var(object):
    @classmethod
    def create_var(cls, var_shape=None, var_dtype=None, val_type=None, var_id=None, device=None, requires_grad=True):
        global val_seq
        vid = var_id
        if not vid:
            vid = val_seq
            val_seq += 1
        var_shape = list(var_shape)
        return Var(vid, val_type, var_shape, var_dtype, device, requires_grad)

    @classmethod
    def copy(cls, other):
        return Var(other.id[1:], other.val_type, other.var_shape, other.var_dtype, other.device, other.requires_grad)

    def __init__(self, var_id, val_type, var_shape, var_dtype, device, requires_grad):
        ''' _users: recording the statments that use this var as inputs
            _stmt: the statement that create this var as output
            _var_type: var type of this var e.g. 
        '''
        self._users = {} 
        self._stmt = None
        self._id = var_prefix + str(var_id)
        self._val_type = val_type
        self._var_shape = var_shape
        self._var_dtype = var_dtype
        self.dtype_str = 'float' if 'float' in str(self._var_dtype).lower() else 'int'
        self._device = device
        self._requires_grad = requires_grad
        self._grad = None
        self._is_grad_of=set()

    def add_user(self, stmt):
        if stmt in self._users:
            self._users[stmt] += 1
        else:
            self._users[stmt] = 1
    
    def set_grad(self, other_var):
        self._grad = other_var
        other_var.set_to_be_grad_of(self)

    def set_to_be_grad_of(self, other_var):
        self._is_grad_of.add(other_var)
    
    def replace_grad(self, new_grad):
        self._grad._is_grad_of.remove(self)
        self._grad = new_grad
        new_grad.set_to_be_grad_of(self)

    def rmv_user(self, stmt):
        self._users[stmt] -= 1
        if self._users[stmt] == 0:
            self._users.pop(stmt, None)
    
    def used_by(self, stmt):
        return stmt in self._users
    
    def detach_from_stmt(self):
        global unused_ids
        unused_ids.add(self.int_id)
        self._stmt = None

    def replace_all_uses_with(self, other_var, propogate_shape):
        for stmt in self._users.keys():
            stmt.replace_arg_with(self, other_var, propogate_shape)
            other_var.add_user(stmt)
        self._users.clear() 
        if len(self._is_grad_of) > 0:
            for var in self._is_grad_of:
                var.set_grad(other_var)
            self._is_grad_of.clear()
    
    def is_srcvar(self):
        return self.val_type == ValType.S
    
    def is_dstvar(self):
        return self.val_type == ValType.D

    def is_edgevar(self):
        return self.val_type == ValType.E

    def is_nodevar(self):
        return self.is_srcvar() or self.is_dstvar()
    
    def __str__(self):
        return str(self._id) + "(" + str(self._val_type) + "," + str(self._var_shape)  +",grad:" + str(self._grad) + ")"

    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._id == other._id
    
    def __hash__(self):
        return hash(self._id) 
    
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id

    @property
    def int_id(self):
        return int(self._id.split(var_prefix)[1])

    @property
    def users(self):
        return self._users

    @property
    def var_shape(self):
        return self._var_shape

    @var_shape.setter
    def var_shape(self, new_shape):
        self._var_shape = new_shape

    @property
    def stmt(self):
        return self._stmt

    @stmt.setter
    def stmt(self, other):
        self._stmt = other

    @property
    def val_type(self):
        return self._val_type

    @property
    def var_dtype(self):
        return self._var_dtype

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self):
        return self._requires_grad

class Stmt(object):
    StmtInfo = namedtuple('StmtInfo', ['op_schema', 'args'])
    @classmethod
    def create_stmt(cls, op_schema=None, args=None, ret=None, callback=None):
        op_type = None
        if ret:
            op_type = infer_op_type_from_args(op_schema, args) 
            ret._requires_grad = any([arg.requires_grad for arg in args if not is_const_scalar(arg)])
        st = Stmt(op_schema, args, ret, op_type, callback)
        return st
    
    @classmethod
    def create_binary_bcast_stmt(cls, op_schema, args, callback=None):
        return Stmt(op_schema, args,
                    Var.create_var(bcast_dim(args), any_var(args).var_dtype, infer_val_type(args), device=any_var(args).device),
                    infer_op_type_from_args(op_schema, args),
                    callback)

    @classmethod
    def create_mul_stmt(cls, args):
        ret_var = Var.create_var(var_shape=args[0].var_shape, var_dtype=args[0].var_dtype, val_type=infer_val_type(args), device=args[0].device)
        st = Stmt.create_stmt(Schema('mul'), args=args, ret=ret_var)
        return st
    
    @classmethod
    def create_add_stmt(cls, args):
        ret_var = Var.create_var(var_shape=args[0].var_shape, var_dtype=args[0].var_dtype, val_type=infer_val_type(args), device=args[0].device)
        st = Stmt.create_stmt(Schema('add'), args=args, ret=ret_var)
        return st

    def __init__(self, op_schema, args, ret, op_type, callback):
        self.op_schema = op_schema
        self.ret = ret
        self.op_type = op_type
        self.callback = register_or_look_up_backend_cb(self, callback)
        self.next = None
        self.prev = None
        self.args = None
        self.op_impl = look_up_registry(self)
        if self.op_impl:
            self.op_impl = self.op_impl(self, Var.create_var, Stmt.create_stmt)
        if args:
            assert isinstance(args, list), 'args must be a list of vars'
            for var in args:
                if isinstance(var, Var):
                    var.add_user(self)
            self.args = args
        if ret:
            ret.stmt = self

    def is_nodewise(self):
        return self.op_type in {OpType.S, OpType.D} 

    def is_edgewise(self):
        return self.op_type == OpType.E
    
    def is_agg(self):
        return self.op_type == OpType.A
    
    def is_src(self):
        return self.op_type == OpType.S

    def is_dst(self):
        return self.op_type == OpType.D

    def depends_on(self, stmt):
        if self == stmt:
            return True
        else:
            return any([arg.stmt.depends_on(stmt) for arg in self.args if not is_const_scalar(arg) and arg.stmt])
    
    def stmt_info(self):
        arg_list = []
        for a in self.args:
            arg = str(a) if is_const_scalar(a) else a.id
            if arg.endswith(inb_attr_postfix):
                arg_list.append(arg[:-len(inb_attr_postfix)])
            elif arg.endswith(cen_attr_postfix):
                arg_list.append(arg[:-len(cen_attr_postfix)])
            else:
                arg_list.append(arg)
        # For node-wise op, the return type can be safely ignored as they are referring to the same tensor
        ret = str(self.op_schema) + '-' + str(arg_list) + (str(self.ret.val_type) if not self.is_nodewise() else '')
        return ret
    
    def is_element_wise_fusable(self):
        if self.op_impl:
            # currently all implemented ops are element-wise except agg ops
            return 'agg' not in self.op_name.lower() and 'gtypecast' not in self.op_name.lower()
        return False
    
    def is_supported(self):
        if self.op_impl:
            return True 
        return False
    
    def copy(self):
        st = Stmt.create_stmt(self.op_schema, args=[Var.copy(arg) if not is_const_scalar(arg) else arg for arg in self.args], ret=Var.copy(self.ret), callback=self.callback)
        return st

    def remove_cur(self):
        if self.args:
            for var in self.args:
                if isinstance(var, Var):
                    var.rmv_user(self)
        if self.ret:
            self.ret.detach_from_stmt()
        self.prev.next = self.next
        self.next.prev = self.prev
    
    def insert_after(self, new_stmt):
        new_stmt.next = self.next
        new_stmt.prev = self
        if self.next:
            self.next.prev = new_stmt
        self.next = new_stmt

    def insert_before(self, new_stmt):
        new_stmt.next = self
        new_stmt.prev = self.prev
        if self.prev:
            self.prev.next = new_stmt
        self.prev = new_stmt
    
    def replace_arg_with(self, old, new, propogate_shape):
        for i in range(len(self.args)):
            if self.args[i] == old:
                self.args[i] = new
                if propogate_shape:
                    if self.is_element_wise_fusable() or 'agg' in self.op_name.lower():
                        self.shape_propogation()
    
    def shape_propogation(self):
        dims = bcast_dim(self.args)  
        if dims != self.ret.var_shape:
            self.ret.var_shape = dims
            for stmt in self.ret.users:
                stmt.shape_propogation()
    
    def __eq__(self, other):
        '''Stats are defined to be equal if they share the same schema, have the same arguments and op_type as well as their ret  sharing the same val_type'''
        return isinstance(other, self.__class__)  and self.op_schema == other.op_schema and self.args == other.args and self.op_type == other.op_type and self.type_eq(self.ret, other.ret)
        
    def type_eq(self, var1, var2):
        return (var1 == None and var2 == None) or (var1.val_type == var2.val_type)
    
    def __hash__(self):
        '''The simpliest implementation as it's unlikely become the bottleneck'''
        return 0

    def __str__(self):
        return str(self.op_type) + ':' +str(self.ret) + '=' + str(self.op_name)  + '(' + str(self.args) + ')'
    
    def __repr__(self):
        return str(self)
    '''Setters and getters'''

    @property
    def op_name(self):
        return self.op_schema._op_name

    @op_name.setter
    def op_name(self, val):
        self.op_schema._op_name = val
    
    def grad(self, y, grad_y):
        return dict(self.op_impl.grad(y, grad_y)) 
    
    def gen_code(self, ctx):
        return self.op_impl.gen_code(ctx)

    def execute(self, args, **kargs):
        return self.callback(*args, **kargs)

class Program(object):
    def __init__(self):
        self.head = Stmt(Schema('head'), None, None, None, None)
        self.tail= Stmt(Schema('tail'), None, None, None, None)
        self.head.next, self.tail.prev = self.tail, self.head
        self.seen_var = {}
    
    def begin(self):
        if self.head.next != self.tail:
            return self.head.next
        else:
            return None
    def end(self):
        if self.tail.prev != self.head:
            return self.tail.prev
        else:
            return None
    
    def empty(self):
        return self.head.next == self.tail
    
    def resort_vars(self):
        global unused_ids
        prev_ret = 0
        available_ids = None
        for s in self:
            if s.ret.int_id < prev_ret:
                max_id = s.ret.int_id
                prev_s = None
                cur_s = None
                for st in self:
                    cur_s = st
                    if st.ret.int_id > max_id:
                        break
                    else:
                        prev_s = st
                min_id = prev_s.ret.int_id
                available_ids = sorted([i for i in unused_ids if i > min_id and i < max_id])
                id_seq = 0
                for s in self:
                    if s.ret.int_id > max_id:
                        if id_seq >= len(available_ids):
                            raise NotImplementedError('Have not considered the case when available ids are not enough')
                        unused_ids.add(s.ret.int_id)
                        unused_ids.remove(available_ids[id_seq])
                        s.ret.id = var_prefix +str(available_ids[id_seq])
                        id_seq += 1
                    elif s.ret.int_id == max_id:
                        break
            prev_ret = s.ret.int_id
    
    def clear_stmts(self):
        self.head.next, self.tail.prev = self.tail, self.head
        self.seen_var = {}

    def input_vars(self):
        ret_set = set()
        input_var_set = set()
        for stmt in self:
            for arg in stmt.args:
                if not is_const_scalar(arg) and arg not in ret_set:
                    input_var_set.add(arg)
            ret_set.add(stmt.ret)
        return input_var_set

    def set_input(self, var_name):
        self.head.op_name = 'input'
        self.head.ret = var_name
    
    def last_stmt(self):
        return self.tail.prev if self.tail.prev != self.head else None

    def append_stmt(self, stmt):
        self.tail.insert_before(stmt)
        self._record_var_in_stmt(stmt)
    
    def prepend_stmt(self, stmt):
        self.head.insert_after(stmt)
        self._record_var_in_stmt(stmt)
    
    def has_stmt(self, stmt):
        for st in self:
            if st == stmt:
                return True
        return False
    
    def deepcopy(self):
        seen_var = {}
        prog = Program()
        for stmt in self:
            prog.tail.insert_before(self._copy_stmt(stmt, seen_var))
        return prog
    
    def copy_append_stmt(self, stmt):
        self.tail.insert_before(self._copy_stmt(stmt, self.seen_var))
        return self

    def copy_prepend_stmt(self, stmt):
        self.head.insert_after(self._copy_stmt(stmt, self.seen_var))
        return self

    def copy_append_stmts(self, stmts):
        for stmt in stmts:
            self.copy_append_stmt(stmt)
        return self

    def copy_append_prog(self, other_prog):
        for stmt in other_prog:
            self.copy_append_stmt(stmt)
        return self

    def copy_prepend_prog(self, other_prog):
        for stmt in reversed(other_prog):
            self.copy_prepend_stmt(stmt)
        return self

    def insert_stmts_before(self, stmt, stmts_list):
        for s in reversed(stmts_list):
            print('current s', s, 'current stmt', stmt)
            stmt.insert_before(s)
            stmt = s

    def find_ret_var_by_id(self, targ_id):
        for stmt in self:
            if stmt.ret.id == targ_id:
                return stmt.ret
        return None
    
    def find_var_by_id(self, targ_id):
        for stmt in self:
            if stmt.ret.id == targ_id:
                return stmt.ret
            for arg in stmt.args:
                if not is_const_scalar(arg) and arg.id == targ_id:
                    return arg
        return None

    def _copy_stmt(self, stmt, seen_var):
        arg_cp = []
        for arg in stmt.args:
            if is_const_scalar(arg):
                arg_cp.append(arg)
            else:
                if not arg.id in seen_var:
                    seen_var[arg.id] = Var.copy(arg)
                arg_cp.append(seen_var[arg.id])
        ret_cp = Var.copy(stmt.ret)
        seen_var[stmt.ret.id] = ret_cp 
        return Stmt.create_stmt(op_schema=stmt.op_schema, args=arg_cp, ret=ret_cp, callback=stmt.callback)

    def _record_var_in_stmt(self, stmt):
        for arg in stmt.args:
            if not is_const_scalar(arg) and arg.id not in self.seen_var:
                self.seen_var[arg.id] = arg
        if stmt.ret.id not in self.seen_var:
            self.seen_var[stmt.ret.id] = self.seen_var

    def __str__(self):
        ret = str(self.head) + '\n'
        for s in self:
            ret += str(s) + '\n'
        return ret
    
    def __repr__(self):
        return str(self)
    
    def __len__(self):
        c = 0
        for _ in self:
            c += 1
        return c

    def __iter__(self):
        '''Allows adding/removing at current nodes while iterating through list'''
        h = self.head.next
        while h != self.tail:
            yield h
            h = h.next

    def __reversed__(self):
        '''Allows adding/removing at current nodes while reversely iterating through list'''
        t = self.tail.prev
        while t != self.head:
            yield t
            t = t.prev
