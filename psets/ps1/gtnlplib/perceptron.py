from collections import defaultdict
from gtnlplib.clf_base import predict,make_feature_vector,argmax

def perceptron_update(x,y,weights,labels):
    """compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """
    y_hat, scores = predict(x, weights, labels)
    update = defaultdict(float)
    if not y_hat == y:
        update.update(make_feature_vector(x,y))
        negative_update = make_feature_vector(x,y_hat)
        for pair, val in negative_update.iteritems():
            negative_update[pair] = -1.0 * val
        update.update(negative_update)
    return update


def estimate_perceptron(x,y,N_its):
    """estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """
    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in xrange(N_its):
        for x_i,y_i in zip(x,y):
            update_amount = perceptron_update(x_i,y_i,weights,labels)
            for pair, val in update_amount.iteritems():
                weights[pair] += float(val)
        weight_history.append(weights.copy())
    return weights, weight_history

def estimate_avg_perceptron(x,y,N_its):
    """estimate averaged perceptron classifier

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """
    labels = set(y)
    w_sum = defaultdict(float) #hint
    weights = defaultdict(float)
    avg_weights = defaultdict(float)
    weight_history = []
    
    t=1.0 #hint
    for it in xrange(N_its):
        for x_i,y_i in zip(x,y):
            curr_update_amount = perceptron_update(x_i, y_i, weights, labels)
            for pair, val in curr_update_amount.iteritems():
                weights[pair] += float(val)
                w_sum[pair] += t * float(val)
            t += 1.0
        avg_weights = weights.copy()
        for pair, val in w_sum.iteritems():
            avg_weights[pair] -= (val / t)
        weight_history.append(avg_weights.copy())
    return avg_weights, weight_history
