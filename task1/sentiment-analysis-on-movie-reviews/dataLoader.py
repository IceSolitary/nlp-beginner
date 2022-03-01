import os
import numpy as np
import pandas as pd
from feature_Extraction import BoW, Ngram
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit, train_test_split


def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu) / sigma;
    return x;


def getData(file="train.tsv/train.tsv", attr='train_and_test', vocabulary={}, feats="Bow"):
    dataset = pd.read_csv(file, sep='\t', engine='python', encoding='ISO-8859-1')

    phrase_ = dataset['Phrase'].values
    phrase = np.array(phrase_)
    if attr == 'train_and_test' or attr == 'train':
        sentiment_ = dataset['Sentiment'].values
        sentiment = np.array(sentiment_)

    if feats == "Ngram":
        ngram = Ngram(vocabulary)
        # phrase_vec, dims = bow.get_Bow(phrase)
        phrase_vec, vocab = ngram.sklearn_get_Ngram(phrase)

        phrase_vec_ = preprocessing.normalize(phrase_vec, axis=1)

    else:
        bow = BoW(vocabulary)
        # phrase_vec, dims = bow.get_Bow(phrase)
        phrase_vec, vocab = bow.sklearn_get_Bow(phrase)

        phrase_vec_ = preprocessing.normalize(phrase_vec, axis=1)

    if attr == 'train_and_test':
        phrase_vec_train, phrase_vec_test, sentiment_train, sentiment_test = train_test_split(phrase_vec, sentiment,
                                                                                              test_size=0.2,
                                                                                              random_state=5)
        return phrase_vec_train, phrase_vec_test, sentiment_train, sentiment_test, vocab
    elif attr == 'train':
        return phrase_vec_, sentiment, vocab
    else:
        return phrase_vec_, vocab
