import logging
from src import dh 
from src import mrf


_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)
_LOGGER.addHandler(logging.StreamHandler())


model = mrf.PaperMRF(
    papers=dh.PAPER_SET,
    n_topics=dh.N_TOPICS,
    references=dh.REFERENCES,
    labels=dh.PAPER_TOPIC_LABELS,    
)
inferer = model.get_inferer()
#graph = inferer.graph