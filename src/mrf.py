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
_LOGGER.setLevel(logging.INFO)

class PaperMRF:
    def __init__(self,
        n_papers: int,
        n_topics: int,
        references: Mapping[int, Set[int]],
        labels: Mapping[PaperId, TopicId],
        is_directional: bool=True
    ):
        """
        """

        self.n_papers = n_papers
        self.n_topics = n_topics
        self.references = references
        self.labels = labels
        self.is_directional = is_directional

        #model parameters
        self.unary_parameters = Variable(torch.rand(n_topics), requires_grad=True)
        self.reference_parameters = Variable(torch.rand(n_topics, n_topics), requires_grad=True)
        
    def run_EM_algorthm(self, n_epochs):
        _LOGGER.debug("running EM for {} epochs".format(n_epochs))
        optimizer = optim.SGD([self.unary_parameters, self.reference_parameters], lr=0.1)
        for i in range(n_epochs):
            optimizer.zero_grad()
            neg_q_func = self._create_negative_q_function()
            _LOGGER.debug("epoch: {},q_value: {}".format(i, neg_q_func.data.numpy()))
            #_LOGGER.debug(self.unary_parameters)
            neg_q_func.backward()
            optimizer.step()
            

    def get_inferer(self):
        return mrf_inf.FactorGraphMRFInference(self.n_papers, self.n_topics,  
            self.references, self.labels, self.unary_parameters.data.numpy(), 
            self.reference_parameters.data.numpy(), self.is_directional)
        
    def _create_negative_q_function(self):
        #getting old values of parameters

        log_unary_params = torch.log(self.unary_parameters)
        log_reference_params = torch.log(self.reference_parameters)

        q_function = 0
        inferer = self.get_inferer()

        #adding unary factors
        for paper_id in range(self.n_papers):
            if paper_id in self.labels:
                q_function += log_unary_params[self.labels[paper_id]]
            else:
                z_prob = torch.tensor(inferer.get_marginals(paper_id), dtype=torch.float32)
                q_function += torch.sum(torch.mul(z_prob, log_unary_params))

        #adding edge factors
        references = self.references
        labels = self.labels
        for i in range(self.n_papers):
            if i in references:
                for j in references[i]:
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
