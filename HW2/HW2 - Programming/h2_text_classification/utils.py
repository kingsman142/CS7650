from nltk.tokenize import regexp_tokenize
import numpy as np

from nltk.corpus import stopwords

# Here is a default pattern for tokenization, you can substitue it with yours
default_pattern =  r"""(?x)
                        (?:[A-Z]\.)+
                        |\$?\d+(?:\.\d+)?%?
                        |\w+(?:[-']\w+)*
                        |\.\.\.
                        |(?:[.,;"'?():-_`])
                    """

def tokenize(text, pattern = default_pattern):
    """Tokenize senten with specific pattern

    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"

    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})

    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    text = text.lower()
    return regexp_tokenize(text, pattern)


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass
    def transform_list(self, text_set):
        pass



class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {}

    def fit(self, text_set: list):
        """Fit a feature extractor based on given data

        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue

    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()

        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]

        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.unigram))
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                feature[self.unigram[text[i].lower()]] += 1

        return feature

    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()

        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]

        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))

        return np.array(features)


class BigramFeature(FeatureExtractor):
    """Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self):
        self.bigram = {}

    def fit(self, text_set):
        """Fit a feature extractor based on given data

        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i, sentence in enumerate(text_set): # iterate over each sentence
            for j, token in enumerate(sentence): # iterate over each sentence's tokens
                token_lowercase = token.lower()
                token_prev_lowercase = sentence[j-1].lower() if j > 0 else "*"
                if (token_lowercase, token_prev_lowercase) not in self.bigram:
                    self.bigram[(token_lowercase, token_prev_lowercase)] = index
                    index += 1
                else:
                    continue

    def transform(self, text):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()

        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]

        Returns:
            array -- an bigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.bigram))
        for i, token in enumerate(text):
            token_lower = token.lower()
            token_prev_lower = text[i-1].lower() if i > 0 else "*"
            if (token_lower, token_prev_lower) in self.bigram:
                feature[self.bigram[(token_lower, token_prev_lower)]] += 1

        return feature

    def transform_list(self, text_set):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()

        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]

        Returns:
            array -- bigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))

        return np.array(features)

class CustomFeature(FeatureExtractor):
    """customized feature extractor, such as TF-IDF
    """
    def __init__(self):
        self.unigram = {}
        self.bigram = {}
        self.stopwords = stopwords.words('english')

    def preprocess_word(self, token):
        #if token in ["male", "female", "he", "she", "her", "him", "woman", "man", "guy", "girl", "lady", "gentleman"]: # marginal increase in accuracy (naive bayes) and decrease (logreg)
        #    token = "GEND"
        #if token in self.stopwords: # decrease in accuracy (naive and logreg)
        #    token = "STOPWORD"
        return token

    def fit(self, text_set: list):
        """Fit a feature extractor based on given data

        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """

        # Replace words in bottom Nth percentile of counts with "UNK"
        '''# get counts of each word
        counts = {}
        for sentence in text_set:
            for word in sentence:
                word = word.lower()
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1
        # prune words in bottom 10th percentile
        index = 0
        counts_values = list(counts.values())
        percentile_n = int(np.percentile(counts_values, q = 25)) # find the count that's at the 25th percentile
        for word in list(counts.keys()):
            count = counts[word]
            if count <= percentile_n:
                counts.pop(word) # remove this word from the dictionary
            else:
                self.unigram[word] = index # valid word, so add it to our unigram dictionary
                index += 1
        self.unigram["UNK"] = index'''

        # Calculate all the TF-IDF weights
        '''# get counts of each word
        counts = {}
        for sentence in text_set:
            sentence_words = set()
            for word in sentence:
                word = word.lower()
                sentence_words.add(word)
                if word in counts:
                    counts[word] = (counts[word][0] + 1, counts[word][1])
                else:
                    counts[word] = (1, 0)
            for word in sentence_words:
                counts[word] = (counts[word][0], counts[word][1]+1)
        # count the number of total words through all the sentences
        total_counts = 0
        for tup in list(counts.values()):
            count, docs = tup
            total_counts += count
        # find the TF-IDF weight of each token
        self.tfidf = {}
        for word, data in counts.items():
            count, docs = data
            tf = count / total_counts
            idf = np.log10(len(text_set) / docs)
            self.tfidf[word] = tf*idf
        # enter all words into the unigram dictionary
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                text_lower = text_set[i][j].lower()
                if text_lower not in self.unigram:
                    self.unigram[text_lower] = index
                    index += 1
                else:
                    continue'''

        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                text_lower = text_set[i][j].lower()
                text_lower = self.preprocess_word(text_lower)
                text_prev_lower = self.preprocess_word(text_set[i][j-1].lower()) if j > 0 else "*"
                if text_lower not in self.unigram:
                    self.unigram[text_lower] = index
                    index += 1
                if (text_lower, text_prev_lower) not in self.bigram:
                    self.bigram[(text_lower, text_prev_lower)] = index
                    index += 1
                else:
                    continue

    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()

        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]

        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        # Used for the unigram + bigram feature
        feature = np.zeros(len(self.unigram) + len(self.bigram))
        for i in range(0, len(text)):
            text_lower = text[i].lower()
            text_lower = self.preprocess_word(text_lower)
            text_prev_lower = self.preprocess_word(text[i-1].lower()) if i > 0 else "*"
            if text_lower in self.unigram:
                feature[self.unigram[text_lower]] += 1
            if (text_lower, text_prev_lower) in self.bigram:
                feature[self.bigram[(text_lower, text_prev_lower)]] += 1

        return feature

    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()

        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]

        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []

        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))

        return np.array(features)
