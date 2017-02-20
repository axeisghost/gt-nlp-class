# 3.1 (0.5 points)

Fill in the rest of the table below:

|      | they | can | can | fish | END |
|------|------|-----|-----|------|-----|
| Noun | -2   | -10 | -10 | -15  | n/a |
| Verb | -13  | -6  | -11 | -16  | n/a |
| End  | n/a  | n/a | n/a | n/a  | -17 |


# 4.3 (0.5 points)

Do you think the predicted tags "PRON AUX AUX NOUN" for the sentence "They can can fish" are correct? Use your understanding of parts-of-speech from the notes.

No, I do not think the predicted tags are correct.
With my understanding of the note, the first "can" should be modals verb, so it should be classified as VERB here. For the second "can", its functionality in the sentence also indicate it should be a VERB but not just auxiliary verb.
# 4.4 (0.5 points)

The HMM weights include a weight of zero for the emission of unseen words. Please explain:

- why this is a violation of the HMM probability model explained in the notes;
  The zero weights of unseen words means that the log-probability of unseen words in the emission model is 1.0. In this way, the probability of all emitted words from a same tag will not sum up to 1. It then violates the emission model of HMM.
- How, if at all, this will affect the overall tagging.
  This actually will eliminate the effect of emission model in the case of unseen words tagging because each tag's emission weights will give zero to the score and then the tagging will only consider the score from the transition model.

# 5.1 (1 point 4650; 0.5 points 7650)

Please list the top three tags that follow verbs and nouns in English and Japanese.
English: DET (determiner), ADP (adposition), PRON (pronoun)
Japanese: NOUN (noun), --END-- (end tag), PUNCT (punctuation)

Try to explain some of the differences that you observe, making at least two distinct points about differences between Japanese and English.

1. English tends to use Verb + noun/pronoun structure in the sentence, but more commonly, English users will put determiner (a, the, this, etc.) between verb and noun to show a more specific indication of the object. The sentence before this can a be good example. However, Japanese does not use verb + noun commonly. For instead, Japanese will put objective first then use particles to connect verb and noun (ゲームをする、晩御飯を食べる、etc) and then the verbs are usually the end of the sentence. That is also the reason why end tags and punctuation will commonly come after verb. 
2. English tends to put adposition word (verb particles) after verb to enrich the meaning of verb (give up, put on, show off, etc.). Japanese, however, tends to combine different verb to one single verb to enrich the meaning of verb. Japanese uses adposition word in different way and most of the time, adposition will not follow verb but follow noun or pronoun.
3. Japanese commonly uses verb to describe noun. It is similar to the attribute in English, but in english the verb usually goes after noun. (event going on, people moving, professor talking, etc.) Same usage in Japanese will put verb before noun and that is the reason why noun commonly comes after verb in Japanese but not in English.

# 6 (7650 only; 1 point)

Find an example of sequence labeling for a task other than part-of-speech tagging, in a paper at ACL, NAACL, EMNLP, EACL, or TACL, within the last five years (2012-2017). 

Answer Extraction as Sequence Tagging with Tree Edit Distance
Xuchen Yao, Benjamin Van Durme, Chris Callison-Burch and Peter Clark
NAACL HLT 2013

## What is the task they are trying to solve?
They are trying extract answers of a question from pretrieved sentences. One process in their pipeline is to tag the sentence with tags related to answer attributes (B-Beginning of the answer, I-Inside the answer, O-Outside the answer). Therefore, they cast the answer finding problem into a squence labeling problem.

## What tagging methods do they use? HMM, CRF, max-margin markov network, something else?
They used CRF with some modification. The final score of a word combines the overlap vote and forced vote. They also apply Median Absolute Deviation to catch outliners words that have different marginal probability than other. These outliners could be possible answers of the question.


## Which features do they use?
1. POS/NER/DEP tags for chunks of words
2. Type of question and answer pairs 
3. The edit distance of a word in their Tree Edit Distance model.
4. The alignment of the words in the QA pair could also be meaningful, so they encode it into several features.

## What methods and features are most effective?
Their experiment shows that combining all features together will achieve the best prediction. For methods, they also use WordNet Search with CRF model and gain the best score in this task.
