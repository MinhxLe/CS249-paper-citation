"""
mrf contains bidirection and unidirectional MRF code and its learning/inference code
"""
import numpy as np
import torch
from torch.autograd import Variable

class PaperMRF:
    def __init__(self, n_papers,
            references,
            n_topics, 
            is_directional=False):
        """
        """
        self.n_papers = n_papers
        self.n_topics = n_topics

        #weights
        self.params = Variable(torch.rand(n_topics, n_topics))
        
    def train_graph(self, topic_labels):
        pass
    def compute_marginal_distribution(self, , references):
        pass
    def compute_MAP(self):
        pass
