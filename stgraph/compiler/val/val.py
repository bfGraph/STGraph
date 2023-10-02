import abc

class Val(abc.ABC):
    @abc.abstractmethod
    def __init__(self, tensor, id, fprog):
        """
        We store the original tensor and create a new tensor with the same attributes as
        the original one except for its size, which is reduced by the fist dimension (assume
        to be 0-th dim) for node and edge tensor.
        _t : the original tensor
        _v : local tracing tensor
        var: intermediate representation of this val
        """
        
        from stgraph.compiler.val.val_factory import ValFactory 
        
        self._t = tensor
        self._id = id
        self._v = None
        self.var = None
        self.fprog = fprog
        self.val_factory = ValFactory()

    @abc.abstractmethod
    def dtype(self):
        """Return the DType"""

    @abc.abstractmethod
    def layout(self):
        """Return the layout"""

    @abc.abstractmethod
    def size(self):
        """Return the size(i.e. shape)"""

    @abc.abstractmethod
    def requires_grad(self):
        """Return whether requires gradient"""

    @abc.abstractmethod
    def val_type(self):
        """Return ValueType"""

    @abc.abstractmethod
    def backend(self):
        """Return backend system"""

    @abc.abstractmethod
    def backend_key(self):
        """Return key of backend system"""

    @abc.abstractmethod
    def device(self):
        """Return which device is on"""

    @abc.abstractmethod
    def view(self, *args, **kargs):
        """Tensor.view"""

    @abc.abstractmethod
    def __mul__(self, other):
        """self * other"""

    @abc.abstractmethod
    def __add__(self, other):
        """self + other"""

    @abc.abstractmethod
    def __radd__(self, other):
        """Override sum edge aggregation, it starts with 0(int) + Val"""

    @abc.abstractmethod
    def __sub__(self, other):
        """self - other"""

    @abc.abstractmethod
    def __truediv__(self, other):
        """self / other"""

    @abc.abstractmethod
    def __floordiv__(self, other):
        """self // other"""

    @abc.abstractmethod
    def sum(self, *args, **kargs):
        """Tensor.sum()"""

    def __str__(self):
        return str(self.var)

    def __repr__(self):
        return str(self)

    @property
    def v(self):
        return self._v

    @property
    def id(self):
        return self._id