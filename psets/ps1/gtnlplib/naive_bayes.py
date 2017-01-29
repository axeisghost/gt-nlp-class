from gtnlplib.preproc import get_corpus_counts
from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation

import numpy as np
from collections import defaultdict

def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    re = defaultdict(float)
    for features, curr_label in zip(x, y):
        if (curr_label == label):
            for word, val in features.iteritems():
                re[word] += float(val)
    return re


    
def estimate_pxy(x,y,label,smoothing,vocab):
    """Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """
    re = defaultdict(float)
    cnt = 0.0
    for features, curr_label in zip(x, y):
        if (curr_label == label):
            for word, val in features.iteritems():
                re[word] += float(val)
                cnt += float(val)
    cnt += len(vocab) * smoothing
    for w in vocab:
        re[w] = np.log((smoothing + re[w]) / cnt)
    return re
    
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    # labels = set(y)
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

    
    
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    """find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values to try
    :returns: best smoothing value, scores of all smoothing values
    :rtype: float, dict

    """
    dv_labels = set(y_tr)
    max_acc = 0.0
    best_smoother = smoothers[0]
    scores = {}
    for smoother in smoothers:
        nb_param = estimate_nb(x_tr, y_tr, smoother)
        y_hat = clf_base.predict_all(x_dv,nb_param,dv_labels)
        curr_acc = evaluation.acc(y_hat, y_dv)
        scores[smoother] = curr_acc
        if (curr_acc > max_acc):
            best_smoother = smoother
    return best_smoother, scores
