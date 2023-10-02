from .cse import CSE
from .cf import CF
from .dce import DCE
from .fusion import fuse
from .mem_planning import mem_planning

def optimize(prog):
    CF(prog)
    CSE(prog)

def joint_optimize(F, B):
    DCE(F, B)