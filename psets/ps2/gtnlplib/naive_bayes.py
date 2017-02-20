import numpy as np #hint: np.log
import sys
from collections import defaultdict,Counter
from gtnlplib import scorer, most_common,preproc
from gtnlplib.constants import OFFSET

def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    counts_y = defaultdict(float)
    counts_xy_sum = defaultdict(float)
    counts_xy = defaultdict(float)
    vocab = set()
    for features, label in zip(x,y):
        counts_y[label] += 1.0
        for word, val in features.iteritems():
            vocab.update([word])
            counts_xy[(label, word)] += float(val)
            counts_xy_sum[label] += float(val)
    for v in vocab:
        for l in counts_xy_sum.iterkeys():
            counts_xy[(l, v)] = np.log((smoothing + counts_xy[(l, v)]) / (counts_xy_sum[l] + smoothing * float(len(vocab))))
    for l, val in counts_y.iteritems():
        counts_xy[(l, OFFSET)] = np.log(val / float(len(y)))
    return counts_xy

def estimate_nb_tagger(counters,smoothing):
    """build a tagger based on the naive bayes classifier, which correctly accounts for the prior P(Y)

    :param counters: dict of word-tag counters, from most_common.get_tag_word_counts
    :param smoothing: value for lidstone smoothing
    :returns: classifier weights
    :rtype: defaultdict

    """
    sorted_tags = sorted(counters.keys())
    nb_weights = estimate_nb([counters[tag] for tag in sorted_tags], sorted_tags, smoothing)
    tot_sum = 0.0
    for tag in sorted_tags:
        cnt_sum = sum(counters[tag].values())
        tot_sum += cnt_sum
        nb_weights[(tag, OFFSET)] = cnt_sum
    for tag in sorted_tags:
        nb_weights[(tag, OFFSET)] = np.log(nb_weights[(tag, OFFSET)] / tot_sum)
    return nb_weights
