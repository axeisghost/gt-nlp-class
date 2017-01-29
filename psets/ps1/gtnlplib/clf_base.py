from gtnlplib.constants import OFFSET
from heapq import nlargest
import numpy as np

# hint! use this.
argmax = lambda x : max(x.iteritems(),key=lambda y : y[1])[0]

def make_feature_vector(base_features,label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    re = {}
    re[(label, OFFSET)] = 1
    for feature in base_features:
        re[(label, feature)] = base_features[feature]
    return re
    
def predict(base_features,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    scores = dict((label, 0.0) for label in labels)
    for label in scores.iterkeys():
        fv = make_feature_vector(base_features, label)
        for pair, cnt in fv.iteritems():
            if weights.has_key(pair):
                scores[label] += weights[pair] * cnt
    return argmax(scores),scores

def predict_all(x,weights,labels):
    """Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    """
    y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])
    return y_hat

def get_top_features_for_label(weights,label,k=5):
    """Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    """
    h = []
    for pair, val in weights.iteritems():
        if (pair[0] == label):
            h.append((val, pair[1]))
    re = nlargest(k, h)
    for i in range(len(re)):
        re[i] = ((label, re[i][1]), re[i][0])
    return re
    
    
