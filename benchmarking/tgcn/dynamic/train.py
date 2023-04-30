import argparse, time

import torch
import numpy as np

from tgcn import SeastarTGCN

from seastar.graph.dynamic.gpma.GPMAGraph import GPMAGraph
from seastar.graph.dynamic.pcsr.PCSRGraph import PCSRGraph
from seastar.dataset.EnglandCOVID import EnglandCOVID

def main(args):
    
    if torch.cuda.is_available():
        print("🎉 CUDA is available")
    else:
        print("😔 CUDA is not available")
        
    eng_covid = EnglandCOVID()
    
    