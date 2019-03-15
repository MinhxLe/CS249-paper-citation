import numpy as np
from src import mrf_inference

n_papers=3
n_topics=2

inferer = mrf_inference.FactorGraphMRFInference(
    n_papers=n_papers,
    n_topics=n_topics,
    references={0: set([1,2]),1: set([2])},
    labels={0:1},
    unary_factors=np.random.random((n_topics)),
    reference_factors=np.random.random((n_topics, n_topics))
)


marginal = inferer.get_joint_marginals(0,1)