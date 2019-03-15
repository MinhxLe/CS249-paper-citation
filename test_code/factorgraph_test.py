import factorgraph as fg
import numpy as np
import torch


n_topics = 2
n_papers = 10

g = fg.Graph()

un_factor = np.array([.5, .5])
ref_factor = np.array([[1, 0.001], [0.001, 0.001]])

for i in range(n_papers-2):
        g.rv(str(i), 1)
        g.factor([str(i)], potential=un_factor[None, 0])
for i in range(0,n_papers-2,2):
        g.factor([str(i), str(i+1)], potential=ref_factor[0,0,None, None])

g.rv(str(n_papers-2), 2)
g.rv(str(n_papers-1), 2)
g.factor([str(n_papers-2)], potential=un_factor)
g.factor([str(n_papers-1)], potential=un_factor)
g.factor([str(n_papers-2), str(n_papers-1)], potential=ref_factor)

iters, converged = g.lbp(normalize=True)
g.print_rv_marginals()
