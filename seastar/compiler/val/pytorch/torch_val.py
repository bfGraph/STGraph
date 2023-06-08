from seastar.compiler.val.val import Val
from seastar.compiler.utils import ValType, infer_val_type
from seastar.compiler.program import Var, Stmt
from seastar.compiler.schema import Schema

class TorchVal(Val):
    def __init__(self, backend, tensor, val_type, id, fprog, reduce_dim):
        self._val_type = val_type
        self._Bkey, self._th = backend
        self._reduce_dim = int(reduce_dim)
        """IR var relies on reducedim"""
        super().__init__(tensor, id, fprog)
        if reduce_dim:
            self._v = self._t.clone().detach().requires_grad_(False).mean(dim=0)
        else:
            self._v = self._t.clone().detach().requires_grad_(False)
        self.var = Var.create_var(
            self.size,
            self.dtype,
            self.val_type,
            var_id=self._id,
            device=self._t.device,
            requires_grad=self._t.requires_grad,
        )
        self.fprog = fprog

    @property
    def backend(self):
        return (self._Bkey, self._th)

    @property
    def backend_key(self):
        return self._Bkey

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def val_type(self):
        return self._val_type

    @property
    def size(self):
        return list(self._t.size()[self._reduce_dim :])

    @property
    def layout(self):
        return self._t.layout

    @property
    def requires_grad(self):
        return self._t.requires_grad

    @property
    def device(self):
        return self._t.get_device()

    def __mul__(self, other):
        vtype = infer_val_type((self, other))
        if isinstance(other, TorchVal):
            ret_val = self.val_factory.create(
                vtype, self.v * other.v, self.backend, None, self.fprog, False
            )

            def call(arg0, arg1):
                return arg0.__mul__(arg1)

            self.fprog.append_stmt(
                Stmt.create_stmt(
                    Schema("Mul"),
                    args=[self.var, other.var],
                    ret=ret_val.var,
                    callback=call,
                )
            )
        else:
            ret_val = self.val_factory.create(
                vtype, self.v * other, self.backend, None, self.fprog, False
            )

            def call(arg0, arg1):
                return arg0.__mul__(arg1)

            self.fprog.append_stmt(
                Stmt.create_stmt(
                    Schema("Mul"),
                    args=[self.var, other],
                    ret=ret_val.var,
                    callback=call,
                )
            )
        return ret_val

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        vtype = infer_val_type((self, other))
        ret_val = self.val_factory.create(
            vtype, self.v + other.v, self.backend, None, self.fprog, False
        )

        def call(arg0, arg1):
            return arg0.__add__(arg1)

        self.fprog.append_stmt(
            Stmt.create_stmt(
                Schema("Add"),
                args=[self.var, other.var],
                ret=ret_val.var,
                callback=call,
            )
        )
        return ret_val

    def __radd__(self, other):
        # agg_sum, we can omit its callback since we will generate code for it
        assert isinstance(other, int)
        assert self.val_type in (ValType.SRC, ValType.EDGE)
        ret_val = self.val_factory.create(
            ValType.DEST, self.v, self.backend, None, self.fprog, False
        )
        self.fprog.append_stmt(
            Stmt.create_stmt(Schema("AggSum"), args=[self.var], ret=ret_val.var)
        )
        return ret_val

    def __sub__(self, other):
        # TODO: An attempt made by us, uncomment the below which is original
        # raise NotImplementedError("__sub__ Op not supported")
        vtype = infer_val_type((self, other))
        ret_val = self.val_factory.create(
            vtype, self.v - other.v, self.backend, None, self.fprog, False
        )

        def call(arg0, arg1):
            return arg0.__sub__(arg1)

        self.fprog.append_stmt(
            Stmt.create_stmt(
                Schema("Sub"),
                args=[self.var, other.var],
                ret=ret_val.var,
                callback=call,
            )
        )
        return ret_val

    def __truediv__(self, other):
        vtype = infer_val_type((self, other))
        ret_val = self.val_factory.create(
            vtype, self.v / other.v, self.backend, None, self.fprog, False
        )

        def call(arg0, arg1):
            return arg0.__truediv__(arg1)

        self.fprog.append_stmt(
            Stmt.create_stmt(
                Schema("TrueDiv"),
                args=[self.var, other.var],
                ret=ret_val.var,
                callback=call,
            )
        )
        return ret_val

    def __floordiv__(self, other):
        raise NotImplementedError("__floordiv__ Op not supported")

    def sum(self, *args, **kargs):
        ret_val = self.val_factory.create(
            self.val_type,
            self.v.sum(*args, **kargs),
            self.backend,
            None,
            self.fprog,
            False,
        )

        def call(*arg, **new_kargs):
            if not new_kargs:
                return arg[0].sum(*arg[1:], **kargs)
            else:
                return arg[0].sum(*arg[1:], **new_kargs)

        self.fprog.append_stmt(
            Stmt.create_stmt(
                Schema("Sum", **kargs),
                args=[self.var]
                + [arg.var if isinstance(arg, Val) else arg for arg in args],
                ret=ret_val.var,
                callback=call,
            )
        )
        return ret_val

    def view(self, *args, **kargs):
        ret_val = self.val_factory.create(
            self.val_type,
            self.v.view(*args, **kargs),
            self.backend,
            None,
            self.fprog,
            False,
        )

        def call(*arg, **new_kargs):
            # the -1 is needed here but not for v.view because
            # for v.view the inputs is reduced by the batch dimesion.
            # We need the -1 to make it work for executing on original tensor.
            if not new_kargs:
                return arg[0].view(-1, *arg[1:], **kargs)
            else:
                return arg[0].veiw(-1, *arg[1:], **new_kargs)

        self.fprog.append_stmt(
            Stmt.create_stmt(
                Schema("View", **kargs),
                args=[self.var]
                + [arg.var if isinstance(arg, Val) else arg for arg in args],
                ret=ret_val.var,
                callback=call,
            )
        )
        return ret_val