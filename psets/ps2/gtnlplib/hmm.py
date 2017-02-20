from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import START_TAG, TRANS, END_TAG, EMIT, OFFSET
from gtnlplib import naive_bayes, most_common
import numpy as np
from collections import defaultdict

def hmm_features(tokens,curr_tag,prev_tag,m):
    """Feature function for HMM that returns emit and transition features

    :param tokens: list of tokens 
    :param curr_tag: current tag
    :param prev_tag: previous tag
    :param i: index of token to be tagged
    :returns: dict of features and counts
    :rtype: dict

    """
    re = {}
    if not curr_tag == START_TAG and not curr_tag == END_TAG:
        re[(curr_tag, tokens[m], EMIT)] = 1
    re[(curr_tag, prev_tag, TRANS)] = 1
    return re
    

def compute_HMM_weights(trainfile,smoothing):
    """Compute all weights for the HMM

    :param trainfile: training file
    :param smoothing: float for smoothing of both probability distributions
    :returns: defaultdict of weights, list of all possible tags (types)
    :rtype: defaultdict, list

    """
    
    tag_trans_counts = most_common.get_tag_trans_counts(trainfile)
    all_tags = tag_trans_counts.keys()
    all_tags.append(END_TAG)
    words_counts = most_common.get_tag_word_counts(trainfile)
    sorted_tags = sorted(words_counts.keys())
    
    nb_weights = naive_bayes.estimate_nb([words_counts[tag] for tag in sorted_tags], sorted_tags, smoothing)
    re = defaultdict(float)
    for key, val in nb_weights.iteritems():
        if not key[1] == OFFSET:
            re[(key[0], key[1], EMIT)] = val
    for tag in all_tags:
        re[(tag, END_TAG, TRANS)] = -np.inf
    re.update(compute_transition_weights(tag_trans_counts, smoothing))
    return re, all_tags


def compute_transition_weights(trans_counts, smoothing):
    """Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag,TRANS)] and weights

    """

    weights = defaultdict(float)
    all_tags_without_END = trans_counts.keys()
    all_tags = trans_counts.keys()
    all_tags.append(END_TAG)
    dominator = defaultdict(float)
    for curr_tag in all_tags:
        for prev_tag in all_tags_without_END:
            if not curr_tag == START_TAG:
                weights[(curr_tag, prev_tag, TRANS)] += smoothing
                if trans_counts.has_key(prev_tag) and trans_counts[prev_tag].has_key(curr_tag):
                    weights[(curr_tag, prev_tag, TRANS)] += trans_counts[prev_tag][curr_tag]
                dominator[prev_tag] += weights[(curr_tag, prev_tag, TRANS)]
            else:
                weights[(curr_tag, prev_tag, TRANS)] = -np.inf
    for feat in weights.keys():
        if not weights[feat] == -np.inf:
            weights[feat] = np.log(weights[feat] / dominator[feat[1]])
    return weights

            

