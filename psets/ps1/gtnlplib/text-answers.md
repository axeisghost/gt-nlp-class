# Deliverable 1.3

Why do you think the type-token ratio is lower for the dev data as compared to the training data?

(Yes the dev set is smaller; why does this impact the type-token ratio?)

The words and phrases in language are not just random sequence of characters and words in passages usually repeat if passages are in the same language. Therefore, when the size of text dataset becomes larger, there will be more repetition of words too. Then the type-token ratio will also be larger. It is the reason why training data has larger type-token ratio.


# Deliverable 3.5

Explain what you see in the scatter plot of weights across different smoothing values.

Result weights when using smoothing=0.001 will have a larger value for some features. In weights when using smoothing=10, these features also has larger weights but they are not so out-standing like the weights in smoothing-0.001. The reason is that some keywords in the topic will have small features values, so the weights have to be larger in order to consider the significance of those features. When the smoothing is too large, small features value will be smoothed like other features, so the its importance will not be reflected in weights. When the smoothing value is small, small values will not be smoothed out and thes size of their weights will reflect their importance.

# Deliverable 6.2

Now compare the top 5 features for logistic regression under the largest regularizer and the smallest regularizer.
Paste the output into ```text_answers.md```, and explain the difference. (.4/.2 points)

clf_base.get_top_features_for_label(theta_lr_largest,'worldnews',5)
Out[102]:
[(('worldnews', u'russia'), 0.15238179434582572),
 (('worldnews', '**OFFSET**'), 0.15159169645252937),
 (('worldnews', u'ukraine'), 0.13716125304025434),
 (('worldnews', u'plane'), 0.12821311362528562),
 (('worldnews', u'russian'), 0.12428483166042341)]
In [103]:

science
clf_base.get_top_features_for_label(theta_lr_largest,'science',5)
Out[103]:
[(('science', u'research'), 0.12331215784129235),
 (('science', '**OFFSET**'), 0.11489780106416261),
 (('science', u'ebv'), 0.11092335575172059),
 (('science', u'study'), 0.11004485054345244),
 (('science', u'corn'), 0.1098369005311165)]
In [104]:

askreddit
clf_base.get_top_features_for_label(theta_lr_largest,'askreddit',5)
Out[104]:
[(('askreddit', u'i'), 0.095505663433061938),
 (('askreddit', u'my'), 0.085127428874173017),
 (('askreddit', u'*'), 0.085119495766494219),
 (('askreddit', u'try'), 0.083626979741890553),
 (('askreddit', u'one'), 0.072437889267936484)]
In [105]:

clf_base.get_top_features_for_label(theta_lr_largest,'iama',5)
Out[105]:
[(('iama', u'!'), 0.16692808321261768),
 (('iama', '**OFFSET**'), 0.15625556395515852),
 (('iama', u'you'), 0.10193474196495818),
 (('iama', u'your'), 0.097378260185317839),
 (('iama', u'i'), 0.08569558535204902)]
In [106]:

rned
clf_base.get_top_features_for_label(theta_lr_largest,'todayilearned',5)
Out[106]:
[(('todayilearned', u'hr'), 0.09036576440289884),
 (('todayilearned', u"''"), 0.072869501671811346),
 (('todayilearned', u'apple'), 0.068808179941801842),
 (('todayilearned', u'``'), 0.066824703164426125),
 (('todayilearned', u'latin'), 0.064957420070386407)]
In [107]:

'worldnews
clf_base.get_top_features_for_label(theta_lr_smallest,'worldnews',5)
Out[107]:
[(('worldnews', u'russia'), 0.35173027429611731),
 (('worldnews', u'plane'), 0.33929653702158746),
 (('worldnews', u'ukraine'), 0.33449249276556642),
 (('worldnews', u'russian'), 0.31561831703819393),
 (('worldnews', u'ai'), 0.29723311599291224)]
In [108]:

science
clf_base.get_top_features_for_label(theta_lr_smallest,'science',5)
Out[108]:
[(('science', u'research'), 0.33806221372344875),
 (('science', u'study'), 0.30292907164499189),
 (('science', u'ebv'), 0.29695098107817364),
 (('science', u'corn'), 0.2694436469585616),
 (('science', u'evolution'), 0.22752261138310378)]
In [109]:

askreddit
clf_base.get_top_features_for_label(theta_lr_smallest,'askreddit',5)
Out[109]:
[(('askreddit', u'porn'), 0.20638394817830613),
 (('askreddit', u'one'), 0.18865452344783626),
 (('askreddit', u'some'), 0.17260046631410891),
 (('askreddit', u'go'), 0.15842356639569363),
 (('askreddit', u'try'), 0.14945713565235963)]
In [110]:

iama
clf_base.get_top_features_for_label(theta_lr_smallest,'iama',5)
Out[110]:
[(('iama', u'!'), 0.24586885174163214),
 (('iama', u'gun'), 0.23250197213754356),
 (('iama', u'thanks'), 0.21211165450797811),
 (('iama', u'state'), 0.21176761634631636),
 (('iama', u'request'), 0.20128478057500812)]
In [111]:

rned
clf_base.get_top_features_for_label(theta_lr_smallest,'todayilearned',5)
Out[111]:
[(('todayilearned', u'hr'), 0.27103036267056163),
 (('todayilearned', u'apple'), 0.21347132896878651),
 (('todayilearned', u'latin'), 0.20945198209359212),
 (('todayilearned', u'bear'), 0.16111617423438407),
 (('todayilearned', u'women'), 0.14135577149133641)]

The Larger regularizer in linear regression will make the size of weights smaller or smoothier. The top 5 features result also confirm this point.
Since the over-smoothed difference between features, even OFFSET will be ranked as top features in largest regularizer case.
Between two result top features in worldnews, the smallest regularizer ranked "plane" before "ukraine" and include "ai" as top feature. On the other hand, the largest regularizer rank "ukraine" before "plane" and has "country" before "ai". It is easy to see that "country" will appear more than "ai" does. Because a large regularizer limit the size of weights, it cannot has a large weight to emphasize low frequency word ("ai"). "country" with a high frequency is still important and it could has a small weight to show its importance. As a result, "country" was ranked higher than "ai" when regularizer is large.

# Deliverable 7.2

Explain the new preprocessing that you designed: why you thought it would help, and whether it did.

New preprocessing simply excludes some high frequency but meaningless words from token counter. These words are called stop words and nltk has built-in list of stop words. I just used it and exclude those stop words in the preprocessing. It can reduce high frequency words without features so it should enhance the classification. After applying the stop words removal, the accuracy was improved 0.2% from 78% to 78.2%.

Bigrams BoW was also tried to replace frequency of words, but it reduce the accuracy to 67%, so I did not include it in the code but comment it out in preproc.py.

# Deliverable 8

Describe the research paper that you have chosen.
  The paper applied multiple recurrent neural network on e-Commerce data to categorize the item. The paper describe the dataset and introduce the architecture of the network. It also provides details about the training method. The paper compared the performance of its model simple, recurrent neural network and Baysian Network with Bag of Words word vector.
- What are the labels, and how were they obtained?
  The labels are the leaf categories of each item. The labels are manually classified by many human experts employed by the owner of the lab.
- Why is it interesting/useful to predict these labels?
  Automatic item categorization can reduce time and economics cost. The paper also emphasized that the accuracy of the categorization can affect the satisfaction of customers which in terms will affect the revenue of the e-commerce sites.
- What classifier(s) do they use, and the reasons behind their choice? Do they use linear classifiers like the ones in this problem set?
  They used a multiple recurrent neural network (Deep categorization network). For each attribute of the text data, they construct an isolated recurrent neural network for it. Then the output of these networks would feed through a full-connected network.
  They believe separated networks can prevent the ambiguity emerging by concatenating the attribute word sequences. They also concluded that the structure not only allows the word vectors to characterize the category-sensitive semantics but also renders a pretraining process such as word2vec unnecessary.
- What features do they use? Explain any features outside the bag-of-words model, and why they used them.
  The features they used contains the word sequence of item name, brand name, High-level-categories, and Maker. They also used nominal data Shopping Mall ID and Image Signature. Word sequences are directly fed into the RNNs and the output are the vector produced from RNNS. Actually these RNNs act likes a "vectorizer" of word sequence data. The output vector is a replacement of bag of words model. They believe this model can separate out the category-sensitive semantics out of the word sequence and that is the reason they build the RNN for each attribute.
- What is the conclusion of the paper? Do they compare between classifiers, between feature sets, or on some other dimension? 
  The paper concluded that DeepCN(the model they built and trained) can dramatically improves the categorization accuracy compared to conventional BoW-based models using the Bayesian networks. One reason is that when the dataset are imbalanced, DeepCN can still precisely categorize leaf categories in a long tail position. They compared their model with Baysian Network using Bag of words and single RNN. They did not compare between features set and different dimension.
- Give a one-sentence summary of the message that they are trying to leave for the reader.
  They believe DeepCN model can be useful in other text classification problem and they are willing to see the improvement and adjustment of the model.
