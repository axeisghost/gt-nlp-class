import nltk
import pandas as pd
from collections import Counter
import string as str

def tokenize_and_downcase(string,vocab=None):
    """for a given string, corresponding to a document:
    - tokenize first by sentences and then by word
    - downcase each token
    - return a Counter of tokens and frequencies.

    :param string: input document
    :returns: counter of tokens and frequencies
    :rtype: Counter

    """
    bow = Counter()
    sentences = nltk.tokenize.sent_tokenize(string, language='english')
    words = []
    for sent in sentences:
        words += nltk.tokenize.word_tokenize(sent, language='english')
    for word in words:
        bow[word.lower()] += 1
    return bow


### Helper code

def read_data(csvfile,labelname,preprocessor=lambda x : x):
    # note that use of utf-8 encoding to read the file
    df = pd.read_csv(csvfile,encoding='utf-8')
    return df[labelname].values,[preprocessor(string) for string in df['text'].values]

def get_corpus_counts(list_of_bags_of_words):
    counts = Counter()
    for bow in list_of_bags_of_words:
        for key,val in bow.iteritems():
            counts[key] += val
    return counts

### Secret bakeoff code
def custom_preproc(string):
    """for a given string, corresponding to a document, tokenize first by sentences and then by word; downcase each token; return a Counter of tokens and frequencies.

    :param string: input document
    :returns: counter of tokens and frequencies
    :rtype: Counter

    """
    bow = Counter()
    sentences = nltk.tokenize.sent_tokenize(string, language='english')
    words = []
    stop = nltk.corpus.stopwords.words('english') + list(str.punctuation)
    for sent in sentences:
        tmp_words = nltk.tokenize.word_tokenize(sent, language='english')
        # words += nltk.ngrams(tmp_words, 2)
        for word in tmp_words:
            if word not in stop:
                words.append(word)
                
    # bow = Counter(nltk.ngrams(words, 2))
    bow = Counter(words)
    return bow
