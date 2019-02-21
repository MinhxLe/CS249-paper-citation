"""
mrf contains bidirection and unidirectional MRF code and its learning/inference code
"""

import numpy as np
import torch

class FullyConnectedBidirectionalMRF:
    def __init__(self, nodes):
        """
        nodes: (unique) list of node identifiers
        """
        self.nodes = nodes
        

        


    def compute_marginal_distribution(self):
        pass
    def compute_MAP(self):
        pass
