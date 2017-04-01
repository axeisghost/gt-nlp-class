from gtnlplib import tagger_base, constants
from collections import defaultdict

def sp_update(tokens,tags,weights,feat_func,tagger,all_tags):
    """compute the structure perceptron update for a single instance

    :param tokens: tokens to tag 
    :param tags: gold tags
    :param weights: weights
    :param feat_func: local feature function from (tokens,y_m,y_{m-1},m) --> dict of features and counts
    :param tagger: function from (tokens,feat_func,weights,all_tags) --> tag sequence
    :param all_tags: list of all candidate tags
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """
    M = len(tags)
    predicted_labels, _ = tagger(tokens,feat_func,weights,all_tags)
    update = defaultdict(float)
    for i in range(len(predicted_labels)):
        if not predicted_labels[i] == tags[i]:
            negative_update = {}
            pos_update = {}
            if i <= 0:
                pos_update = feat_func(tokens, tags[i], constants.START_TAG, i)
                negative_update = feat_func(tokens, predicted_labels[i], constants.START_TAG, i)
            else:
                pos_update = feat_func(tokens, tags[i], tags[i-1], i)
                negative_update = feat_func(tokens, predicted_labels[i], predicted_labels[i-1], i)
            for pair, val in pos_update.iteritems():
                update[pair] += val
            for pair, val in negative_update.iteritems():
                update[pair] -= val
    # end_update = feat_func(tokens, constants.END_TAG, tags[-1], M)
    # for pair, val in end_update.iteritems():
    #     update[pair] += val
    return update
    
def estimate_perceptron(labeled_instances,feat_func,tagger,N_its,all_tags=None):
    """Estimate a structured perceptron

    :param labeled instances: list of (token-list, tag-list) tuples, each representing a tagged sentence
    :param feat_func: function from list of words and index to dict of features
    :param tagger: function from list of words, features, weights, and candidate tags to list of tags
    :param N_its: number of training iterations
    :param all_tags: optional list of candidate tags. If not provided, it is computed from the dataset.
    :returns: weight dictionary
    :returns: list of weight dictionaries at each iteration
    :rtype: defaultdict, list

    """
    """
    You can almost copy-paste your perceptron.estimate_avg_perceptron function here. 
    The key differences are:
    (1) the input is now a list of (token-list, tag-list) tuples
    (2) call sp_update to compute the update after each instance.
    """

    # compute all_tags if it's not provided
    if all_tags is None:
        all_tags = set()
        for tokens,tags in labeled_instances:
            all_tags.update(tags)

    # this initialization should make sure there isn't a tie for the first prediction
    # this makes it easier to test your code
    weights = defaultdict(float,
                          {('NOUN',constants.OFFSET):1e-3})

    weight_history = []

    # the rest is up to you!
    w_sum = defaultdict(float) #hint
    avg_weights = defaultdict(float)
    # tokens, tags = labeled_instances
    t=0.0 #hint
    for it in xrange(N_its):
        for tokens, tags in labeled_instances:
            curr_update_amount = sp_update(tokens,tags,weights,feat_func,tagger,all_tags)
            for pair, val in curr_update_amount.iteritems():
                weights[pair] += float(val)
                w_sum[pair] += t * float(val)
            t += 1.0
        avg_weights = weights.copy()
        for pair, val in w_sum.iteritems():
            avg_weights[pair] -= (val / t)
        weight_history.append(avg_weights.copy())
    return avg_weights, weight_history



