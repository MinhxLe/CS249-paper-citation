import numpy as np
from pybasicbayes import models

#defining example graph and factors(?)
n_topics = 3
vertices = [0,1,2]
edges = [
        set([2,3]),
        set([3]),
        set([])
        ]


