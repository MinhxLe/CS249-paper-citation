"""
mrf contains bidirection and unidirectional MRF code and its learning/inference code
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import logging

from typing import List, Mapping, Set
from src.types import PaperId, TopicId
from src import mrf_inference as mrf_inf

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)

class PaperMRF:
    def __init__(self,
        papers: Set[PaperId],
        n_topics: int,
        references: Mapping[int, Set[int]],
        labels: Mapping[PaperId, TopicId],
        is_directional: bool=True
    ):
        """
        """

        self.papers = papers
        self.n_topics = n_topics
        self.references = references
        self.labels = labels
        self.is_directional = is_directional

        #model parameters
        self.unary_parameters = Variable(torch.rand(n_topics), requires_grad=True)
        self.reference_parameters = Variable(torch.rand(n_topics, n_topics), requires_grad=True)
        
    def run_EM_algorthm(self, n_epochs, lr=0.1):
        losses = []
        _LOGGER.debug("running EM for {} epochs".format(n_epochs))
        optimizer = optim.SGD([self.unary_parameters, self.reference_parameters], lr=lr)
        for i in range(n_epochs):
            optimizer.zero_grad()
            neg_q_func = self._create_negative_q_function()
            losses.append(neg_q_func.data.numpy())
            _LOGGER.debug("epoch: {},q_value: {}".format(i, neg_q_func.data.numpy()))
            neg_q_func.backward()
            optimizer.step()
        return np.array(losses)

    def get_inferer(self):
        _LOGGER.debug("getting inferer object")
        inferer = mrf_inf.FactorGraphMRFInference(self.papers, self.n_topics,  
            self.references, self.labels, self.unary_parameters.data.numpy(), 
            self.reference_parameters.data.numpy(), self.is_directional)
        self.inferer = inferer
        return inferer
    def _create_negative_q_function(self):
        _LOGGER.debug("getting negative q function")
        #getting old values of parameters

        log_unary_params = torch.log(self.unary_parameters)
        log_reference_params = torch.log(self.reference_parameters)

        q_function = 0
        inferer = self.get_inferer()

        #adding unary factors
        for paper_id in self.papers:
            if paper_id in self.labels:
                q_function += log_unary_params[self.labels[paper_id]]
            else:
                z_prob = torch.tensor(inferer.get_marginals(paper_id), dtype=torch.float32)
                q_function += torch.sum(torch.mul(z_prob, log_unary_params))

        #adding edge factors
        references = self.references
        labels = self.labels
        for i in self.papers:
            if i in references:
                for j in references[i]:
                    assert(j in self.papers)
                    if i in labels and j in labels:
                        q_function += log_reference_params[labels[i], labels[j]]    
                    elif i in labels:
                        z_prob = torch.tensor(inferer.get_marginals(j), dtype=torch.float32)
                        param = log_reference_params[labels[i],:]
                        q_function += torch.sum(torch.mul(z_prob, param))
                    elif j in labels:
                        z_prob = torch.tensor(inferer.get_marginals(i), dtype=torch.float32)
                        param = log_reference_params[:, labels[j]]
                        q_function += torch.sum(torch.mul(z_prob, param))
                    else:
                        z_prob = torch.tensor(inferer.get_joint_marginals(i,j))
        return -q_function
