import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set

        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model

        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions

        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZeor(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        self.fit_called = False
        self.add_k_smoothing = 1 # add-1 smoothing

    def fit(self, X, Y):
        D = X.shape[1]

        # separate our two classes
        hate_speech_indices = (Y == 1)
        hate_speech_samples = X[hate_speech_indices]
        nonhate_speech_samples = X[~hate_speech_indices]

        # count number of times each word occurs in each class
        hate_speech_vocab_counts = hate_speech_samples.sum(axis = 0)
        nonhate_speech_vocab_counts = nonhate_speech_samples.sum(axis = 0)

        # count number of words in each class
        hate_speech_total_counts = np.sum(hate_speech_vocab_counts) + (self.add_k_smoothing * D) # the "+ D" is used for the denominator of add-1 smoothing
        nonhate_speech_total_counts = np.sum(nonhate_speech_vocab_counts) + (self.add_k_smoothing * D) # the "+ D" is used for the denominator of add-1 smoothing

        # divide the count of each word by the total # of words to get the probability of that word occurring
        self.hate_speech_word_probs = np.log(((hate_speech_vocab_counts + self.add_k_smoothing) / hate_speech_total_counts)) # make sure to do add-1 smoothing
        self.nonhate_speech_word_probs = np.log(((nonhate_speech_vocab_counts + self.add_k_smoothing) / nonhate_speech_total_counts)) # make sure to do add-1 smoothing

        # individual class probabilities
        self.hate_prob = np.log(len(hate_speech_samples) / (len(hate_speech_samples) + len(nonhate_speech_samples)))
        self.nonhate_prob = np.log(1 - self.hate_prob)

        self.fit_called = True

    def predict(self, X):
        if not self.fit_called:
            print("You must train/fit the model first!...")
            return None

        N = len(X)
        pred = np.zeros(N) # predictions

        for index, sample in enumerate(X):
            # calculate probability of being a hate statement
            hate_probability = self.hate_prob
            for word_id, word_count in enumerate(sample):
                hate_probability += (word_count * self.hate_speech_word_probs[word_id])

            # calculate probability of being a non-hate statement
            nonhate_probability = self.nonhate_prob #np.log(self.nonhate_prob)
            for word_id, word_count in enumerate(sample):
                nonhate_probability += (word_count * self.nonhate_speech_word_probs[word_id])

            pred[index] = 1 if hate_probability > nonhate_probability else 0

        return pred

# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        self.eta = .02 # learning rate
        self.epochs = 100 # number of epochs
        self.alpha = 0.001 # regularization weight

    def fit(self, X, Y):
        N, D = X.shape
        self.theta = np.random.rand(D)

        train_size = int(N * 0.9)
        X_train = X[0:train_size]
        Y_train = Y[0:train_size]
        X_val = X[train_size:]
        Y_val = Y[train_size:]

        for epoch in range(self.epochs):
            old_theta = self.theta
            avg_loss = 0.0
            for index, sample in enumerate(X):
                label = Y[index]

                sigmoid = 1 / (1 + np.exp(-np.dot(self.theta, sample))) # calculate class probability (sigmoid value)
                gradient = (sigmoid - label) * sample + self.alpha * (2 * self.theta) # calculate gradient of cross-entropy loss + regularization term
                self.theta -= self.eta * gradient # update weights

                loss = -(label * np.log(sigmoid) + (1 - label) * np.log(1 - sigmoid)) + (self.alpha * (np.linalg.norm(self.theta)**2))
                avg_loss += loss
            avg_loss /= N
            if (epoch+1) % 1 == 0:
                print("Epoch {}/{}, Sample {}/{} -- Loss: {}".format(epoch+1, self.epochs, index+1, N, avg_loss))

    def predict(self, X):
        N = len(X)
        pred = np.zeros(N) # predictions

        for index, sample in enumerate(X):
            sample_prob = 1 / (1 + np.exp(-np.dot(self.theta, sample)))
            pred[index] = 1 if sample_prob > 0.5 else 0

        return pred
