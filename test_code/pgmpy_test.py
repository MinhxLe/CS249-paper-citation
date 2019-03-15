import numpy as np
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

G = FactorGraph()
G.add_node(0)
G.add_node(1)
G.add_node(2)


f01 = DiscreteFactor([0,1], [2, 2], np.random.rand(4))
f02 = DiscreteFactor([0,2], [2, 2], np.random.rand(4))
f12 = DiscreteFactor([1,2], [2, 2], np.random.rand(4))
G.add_factors(f01)
G.add_factors(f02)
G.add_factors(f12)

G.add_edges_from([(0, f01), (1, f01), (0, f02), (2, f02), (1, f12), (2, f12)])
bp = BeliefPropagation(G)
bp.calibrate()