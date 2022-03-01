import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


class BoW:

    def __init__(self, vocabulary={}):
        self.vocab = vocabulary

    def construct_dic(self, sent_list):
        for sent in sent_list:
            sent = sent.lower()
            words = sent.strip().split(" ")
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

    def BoW(self, sent_list):
        vec = []

        for sent in sent_list:
            sent_vec = np.zeros(len(self.vocab))
            sent = sent.lower()
            words = sent.strip().split(" ")
            for word in words:
                sent_vec[self.vocab[word]] += 1

            vec.append(sent_vec.tolist())
        return vec

    def get_Bow(self, sent_list):
        self.construct_dic(sent_list)
        return self.BoW(sent_list), len(self.vocab)

    def sklearn_get_Bow(self, sent_list):
        if self.vocab != {}:
            count_vectorizer = CountVectorizer(vocabulary=self.vocab)
        else:
            count_vectorizer = CountVectorizer()
        count_matrix = count_vectorizer.fit_transform(sent_list)
        self.vocab = count_vectorizer.vocabulary_

        return count_matrix, self.vocab


class Ngram:
    def __init__(self, ngram):
        self.vocab = {}
        self.ngram = ngram

    def construct_dic(self, sent_list):
        feature_min = self.ngram[0]
        feature_max = self.ngram[1]

        # 1.0 version
        # for n_gram in range(feature_min, feature_max+1):
        #     for sent in sent_list:
        #         sent = sent.lower()
        #         words = sent.strip().split(" ")
        #         for i in range(len(words) - n_gram + 1):
        #             for j in range(n_gram):
        #                 if j == 0:
        #                     word = words[i]
        #                 if j > 0:
        #                     word = word + " " + words[i+j]
        #         if word not in self.vocab:
        #             self.vocab[word] = len(self.vocab)

        # 1.1 version
        vocab_ = []
        for sent in sent_list:
            sent = sent.lower()
            words = sent.strip().split()
            words = np.char.array(words)
            idx = np.arange(len(words)).reshape(-1, 1)

            ngrams = []

            for n_gram in range(1, 2 + 1):
                gram_id = np.arange(n_gram).reshape(1, -1)
                idx_ = (idx + gram_id)[:len(words) - n_gram + 1, :]

                words_ = words[idx_]
                words_out = words_[:, 0].copy()
                for i in range(1, n_gram):
                    words_out = words_out + " " + words_[:, i]

                ngrams.append(words_out)

            ngrams = np.concatenate(ngrams, 0)
            ngrams = np.unique(ngrams)
            vocab_.append(ngrams)

        vocab_ = np.concatenate(vocab_, 0)
        vocab_ = np.unique(vocab_)
        index = np.arange(len(vocab_))
        self.vocab = {k: v for k, v in zip(ngrams, index)}

    def Ngram(self, sent_list):
        feature_min = self.ngram[0]
        feature_max = self.ngram[1]

        vec = []

        for n_gram in range(feature_min, feature_max + 1):
            for sent in sent_list:
                sent = sent.lower()
                sent_vec = np.zeros(len(self.vocab))
                words = sent.strip().split(" ")
                for i in range(len(words) - n_gram + 1):
                    word = words[i]
                    for j in range(1, i):
                        word = word + " " + words[j]
                    sent_vec[self.vocab[word]] += 1
            vec.append(sent_vec)

    def get_Ngram(self, sent_list):
        self.construct_dic(sent_list)
        return self.ngram(sent_list), len(self.vocab)

    def sklearn_get_Ngram(self, sent_list):
        if self.vocab != {}:
            count_vectorizer = CountVectorizer(vocabulary=self.vocab, ngram_range=(1,3))
        else:
            count_vectorizer = CountVectorizer(ngram_range=(1, 3))
        count_matrix = count_vectorizer.fit_transform(sent_list)
        self.vocab = count_vectorizer.vocabulary_

        return count_matrix, self.vocab
