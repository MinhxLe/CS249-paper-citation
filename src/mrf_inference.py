F"""
inference module defines an inference interface for MRF class to use that essentially
provides wrappers for blackbox libraries such as factorgraph.py
"""
import numpy as np
from typing import List, Mapping, Set
import abc
import logging
from fglib import graphs, nodes, inference, rv

from src.types import PaperId, TopicId
from external import factorgraph as fg

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)

class MRFInference(abc.ABC):
    @abc.abstractmethod
    def __init__(self,
        papers: Set[PaperId],
        n_topics: int,
        references: Mapping[int, Set[int]],
        labels: Mapping[PaperId, TopicId],
        unary_factors,
        reference_factors,
        is_directional: bool=True
    ):
        pass
    @abc.abstractmethod
    def get_marginals(self, node_id):
        pass

    @abc.abstractmethod
    def get_joint_marginals(self, node_id1, node_id2):
        pass
"""
class FGLibMRFInference(MRFInference):
    def __init__(self,
        papers: Set[PaperId],
        n_topics: int,
        references: Mapping[int, Set[int]],
        labels: Mapping[PaperId, TopicId],
        unary_factors,
        reference_factors,
        is_directional: bool=True
    ):
    rvs = {}
    factors = {}
    
    for paper in papers:
        rvs[paper] = nodes.VNode(str(paper), rv.Discrete)
    
    #defining unary factors
    #TODO we can make this more complex by adding word nodes
    for paper in papers:
        if paper in labels:
            factor = nodes.FNode('f({})'.format(paper), rv.Discrete(, rvs[paper]))
            graph.factor([str(i)], potential=unary_factors[None,labels[i]])
        else:
            factor = fc
            graph.factor([str(i)], potential=unary_factors)
        
    def get_marginals(self, node_id):
        pass

    def get_joint_marginals(self, node_id1, node_id2):
        pass
"""
class FactorGraphMRFInference(MRFInference):
    def __init__(self,
        papers: Set[PaperId],
        n_topics: int,
        references: Mapping[int, Set[int]],
        labels: Mapping[PaperId, TopicId],
        unary_factors,
        reference_factors,
        is_directional: bool=True
    ):
        graph = fg.Graph()
        #defining variables
        for i in papers:
            if i in labels:
                graph.rv(str(i), 1)
            else:
                graph.rv(str(i), n_topics)
        #defining unary factors
        #TODO we can make this more complex by adding word nodes
        for i in papers:
            if i in labels:
                graph.factor([str(i)], potential=unary_factors[None,labels[i]])
            else:
                graph.factor([str(i)], potential=unary_factors)
        
        #defining binary factors
        for i in papers:
            if i in references:
                for j in references[i]:
                    # should only occur from self references...which doesn't make sense
                    assert(i != j) 
                    if i in labels and j in labels:
                        potential = reference_factors[labels[i],labels[j], None, None]
                    elif i in labels:
                        potential = reference_factors[None, labels[i],:]
                    elif j in labels:
                        potential = reference_factors[:,labels[j], None]
                    else:
                        potential = reference_factors
                    graph.factor([str(i), str(j)], potential=potential)
        iters, converged = graph.lbp(normalize=True, max_iters=100)
        if not converged:
            _LOGGER.warning("LBP algorithm did not converge!")
        self.graph = graph



    def get_marginals(self, node_id):
        dist =  self.graph.rv_marginals([self.graph._rvs[str(node_id)]])[0][1]
        return dist / np.sum(dist)

    def get_joint_marginals(self, node_id1, node_id2):
        #find the appropiate factor in the graph
        graph = self.graph
        node_id1, node_id2 = str(node_id1), str(node_id2)
        factor = graph._factor_map[tuple([node_id1, node_id2])]

        factor_msg1 = graph._rvs[node_id1].get_outgoing_for(factor)
        factor_msg2 = graph._rvs[node_id2].get_outgoing_for(factor)
    
        dist = np.outer(factor_msg1, factor_msg2)
        dist /= np.sum(dist)

        return dist