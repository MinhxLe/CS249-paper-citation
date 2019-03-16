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
parser.add_argument('--n_epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--is_directional', type=bool, default=True)
parser.add_argument('--n_loopy', type=int, default=100)
args = parser.parse_args()

model_name = "PaperMRF_p={}_lr={}_is_directional={}".format(args.p, args.lr, args.is_directional)

paper_set = dh.PAPER_SET.copy()
references = dh.REFERENCES.copy()
labels = dh.PAPER_TOPIC_LABELS.copy()

test_paper_set = set()
#keeping only paper_set and only p% of them
del_list = []
for key in labels:
    if not key in paper_set:
        del_list.append(key)
    elif np.random.uniform() > args.p:
            del_list.append(key)
            test_paper_set.add(key)
for key in del_list:
    del labels[key]

#removing all papers not in paper_set from references
del_list = []
for key in references:
    if not key in paper_set:
        del_list.append(key)
    references[key].intersection_update(paper_set)

for key in del_list:
    del references[key]


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
    n_loopy=args.n_loopy
)
_LOGGER.debug("training model for {} epochs".format(args.n_epochs))
losses = model.run_EM_algorthm(args.n_epochs, lr=args.lr)

_LOGGER.debug('saving model {}'.format(model_name))
with open(os.path.join('models', model_name + ".pkl"), 'wb') as f:
    pickle.dump(model, f)

_LOGGER.debug("saving losses")
fname = "losses/{}_losses.npy".format(model_name)
np.save(fname, losses)

count = 0
for paper in test_paper_set:
    if np.argmax(model.inferer.get_marginals(paper)) == dh.PAPER_TOPIC_LABELS[paper]:
        count += 1
accuracy = count/len(test_paper_set)

_LOGGER.debug("Final Accuracy: {}".format(accuracy))