from collections import defaultdict
import numpy as np
import pandas as pd
import os
import logging

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)
_LOGGER.addHandler(logging.StreamHandler())

_DATA_PATH = './data'
_TOPIC_LABEL_DICT = {'Information_Retrieval': 0, 'Databases': 1,
                     'Artificial_Intelligence': 2, 'Networking': 3,
                     'Encryption_and_Compression': 4, 'Operating_Systems': 5,
                     'Data_Structures__Algorithms_and_Theory': 6,
                     'Hardware_and_Architecture': 7,
                     'Programming': 8, 'Human_Computer_Interaction': 9}

def _get_references():
    references = defaultdict(set)
    reference_pd = np.genfromtxt(os.path.join(_DATA_PATH, 'citations'), dtype=np.int32)
    for row in reference_pd:
        references[str(row[0])].add(str(row[1]))
    for key in references:
        #removing self references
        if key in references[key]:
            references[key].remove(key)

    return references

def _get_labels():
    paperToIdMap = {}
    with open(os.path.join(_DATA_PATH, 'papers'), 'r') as f:
        for line in f:
            line = line.split()
            paperToIdMap[line[1]] = line[0]



    classification_map = {}
    classifications_df = pd.read_csv(os.path.join(_DATA_PATH, "classifications"), 
        delimiter='\t', header=None)

    for index, row in classifications_df.iterrows():
        if row[0] in paperToIdMap:
            paper_id = paperToIdMap[row[0]]
            topic = row[1].split('/')[1]
            classification_map[paper_id] = _TOPIC_LABEL_DICT[topic]

    return classification_map

def _remove_invalid_paper(topic_labels, references):
    label_set = set([id for id in topic_labels])
    ref_set = set()
    for paper,ref in references.items():
        ref_set.add(paper)
        ref_set.union(ref)
    
    paper_set = label_set.intersection(ref_set)

    del_list = []
    for key in topic_labels:
        if key not in paper_set:
            del_list.append(key)
    
    for key in del_list:
        del topic_labels[key]
    
    del_list= []
    for paper, ref in references.items():
        if not paper in paper_set:
            del_list.append(paper)
        else:
            references[paper].intersection_update(paper_set)
    for key in del_list:
        del references[key]
    
    return paper_set, topic_labels, references

PAPER_SET, PAPER_TOPIC_LABELS, REFERENCES = _remove_invalid_paper(_get_labels(), _get_references())
N_TOPICS = 10
_LOGGER.debug("Finished loading data!")