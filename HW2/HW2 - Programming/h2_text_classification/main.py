import pandas as pd
from classifiers import *
from utils import *
import numpy as np
import time
import argparse


####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


def accuracy(pred, labels):
    correct = (np.array(pred) == np.array(labels)).sum()
    accuracy = correct/len(pred)
    print("Accuracy: %i / %i = %.4f " %(correct, len(pred), correct/len(pred)))


def read_data(path):
    train_frame = pd.read_csv(path + 'train.csv')

    # You can form your test set from train set
    # We will use our test set to evaluate your model
    try:
        test_frame = pd.read_csv(path + 'test.csv')
    except:
        test_frame = train_frame

    return train_frame, test_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AlwaysPredictZero',
                        choices=['AlwaysPredictZero', 'NaiveBayes', 'LogisticRegression'])
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'customized'])
    parser.add_argument('--path', type=str, default = './data/', help='path to datasets')
    args = parser.parse_args()
    print(args)

    train_frame, test_frame = read_data(args.path)

    # Convert text into features
    if args.feature == "unigram":
        feat_extractor = UnigramFeature()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
    elif args.feature == "customized":
        feat_extractor = CustomFeature()
    else:
        raise Exception("Pass unigram, bigram or customized to --feature")

    # Tokenize text into tokens
    tokenized_text = []
    for i in range(0, len(train_frame['text'])):
        tokenized_text.append(tokenize(train_frame['text'][i]))

    feat_extractor.fit(tokenized_text)

    # form train set for training
    X_train = feat_extractor.transform_list(tokenized_text)
    Y_train = train_frame['label']


    # form test set for evaluation
    tokenized_text = []
    for i in range(0, len(test_frame['text'])):
        tokenized_text.append(tokenize(test_frame['text'][i]))
    X_test = feat_extractor.transform_list(tokenized_text)
    Y_test = test_frame['label']


    if args.model == "AlwaysPredictZero":
        model = AlwaysPreditZeor()
    elif args.model == "NaiveBayes":
        model = NaiveBayesClassifier()
    elif args.model == "LogisticRegression":
        model = LogisticRegressionClassifier()
    else:
        raise Exception("Pass AlwaysPositive, NaiveBayes, LogisticRegression to --model")


    start_time = time.time()
    model.fit(X_train,Y_train)
    # OLD: remove the next 6 lines before submission
    '''most_distinct_words_probs = model.hate_speech_word_probs / model.nonhate_speech_word_probs
    most_distinct_words = np.argsort(-most_distinct_words_probs)[-20:]
    most_distinct_probs = [most_distinct_words_probs[id] for id in most_distinct_words]
    new_dict = dict([[v, k] for k, v in feat_extractor.unigram.items()])
    for index, id in enumerate(most_distinct_words):
        print(new_dict[id], most_distinct_probs[index])'''
    print("===== Train Accuracy =====")
    accuracy(model.predict(X_train), Y_train)

    print("===== Test Accuracy =====")
    accuracy(model.predict(X_test), Y_test)

    print("Time for training and test: %.2f seconds" % (time.time() - start_time))



if __name__ == '__main__':
    main()
