class Schema(object):
    '''Schema of an op'''
    def __init__(self, op_name, **kargs):
        self._op_name = op_name
        self._params = kargs

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._op_name == other._op_name and self._params == other._params

    def __str__(self):
        return str(self._op_name) + '('  + str(self._params) +  ')'