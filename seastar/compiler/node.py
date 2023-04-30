from .utils import EdgeDirection
from .program import Stmt
from .val import create_src_node_val
from .schema import Schema

class NbNode(object):
    def __init__(self, center, direction):
        self._central_node = center
        self._direction = direction

class NbEdge(object):
    def __init__(self, center, direction, nbnodes):
        self._direction = direction
        if self._direction == EdgeDirection.IN:
            self.src, self.dst = nbnodes, center
        elif self._direction == EdgeDirection.OUT:
            self.src, self.dst = center, nbnodes

class CentralNode(object):
    def __init__(self):
        self.innbs = [NbNode(self, EdgeDirection.IN)] 
        self.outnbs = [NbNode(self, EdgeDirection.OUT)] 
        self.inedges = [NbEdge(self, EdgeDirection.IN, self.innbs)] 
        self.outedges = [NbEdge(self, EdgeDirection.OUT, self.outnbs)] 
    
    def update_allnode(self, feat_map):
        for k,v in feat_map.items():
            setattr(self, k, v)
            for nb in self.innbs:
                setattr(nb, k, create_src_node_val(v._v, v.backend, id=str(v.var.int_id), fprog=v.fprog, reduce_dim=False))
                nb.__dict__[k].var._stmt = Stmt.create_stmt(op_schema=Schema('GTypeCast'), args=[v.var], ret=nb.__dict__[k].var) 
                v.fprog.append_stmt(nb.__dict__[k].var._stmt)
                nb.__dict__[k].var._requires_grad = v.var._requires_grad