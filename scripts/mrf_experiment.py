import logging
from src import dh 
from src import mrf
import argparse
import pickle
import numpy as np
import os

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)
_LOGGER.addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=float, help="percentage held out")
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--is_directional', type=bool, default=True)

args = parser.parse_args()

model_name = "PaperMRF_p={}_lr={}_is_directional={}".format(args.p, args.lr, args.is_directional)

_LOGGER.debug(model_name)
_LOGGER.debug("holding out {} labels".format(args.p))
labels = dh.PAPER_TOPIC_LABELS.copy()
del_list = []
for key in labels:
    if np.random.uniform() > args.p:
        del_list.append(key)
for key in del_list:
    del labels[key]

model = mrf.PaperMRF(
    papers=dh.PAPER_SET,
    n_topics=dh.N_TOPICS,
    references=dh.REFERENCES,
    labels=labels,    
)
_LOGGER.debug("training model for {} epochs".format(args.n_epochs))
losses = model.run_EM_algorthm(args.n_epochs, lr=args.lr)

_LOGGER.debug('saving model {}'.format(model_name))
with open(os.path.join('models', model_name + ".pkl"), 'wb') as f:
    pickle.dump(model, f)

_LOGGER.debug("saving losses")
fname = "losses/{}_losses.npy".format(model_name)
np.save(fname, losses)