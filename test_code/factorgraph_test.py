import factorgraph as fg
import numpy as np
import torch


n_topics = 3
n_papers = 10

g = fg.Graph()

# Add some discrete random variables (RVs)
g.rv('a', 2)
g.rv('b', 3)

# Add some factors, unary and binary
g.factor(['a'], potential=np.array([0.3, 0.7]))
g.factor(['b', 'a'], potential=np.array([
        [0.2, 0.8],
        [0.4, 0.6],
        [0.1, 0.9],
]))

iters, converged = g.lbp(normalize=True)

