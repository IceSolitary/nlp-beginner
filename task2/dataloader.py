import os
import pickle
import time

import pandas as pd
import spacy
import torch
# from torchtext.legacy.data import Dataset,Field, TabularDataset, Iterator, BucketIterator
from sklearn.model_selection import train_test_split
from torch.nn import init
from torchtext.legacy import data


spacy_en = spacy.load('en_core_web_sm')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def splitData(train_file, test_file):
    train_ = pd.read_csv(train_file, sep='\t', engine='python', encoding='ISO-8859-1')
    test_ = pd.read_csv(test_file, sep='\t', engine='python', encoding='ISO-8859-1')

    train_data, val_data = train_test_split(train_, test_size=0.2)
    print(type(train_data))

    # train_data.to_csv("train.csv", index=False)
    # val_data.to_csv("val.csv", index=False)
    # test_.to_csv("test.csv", index=False)


# splitData("train.tsv", "test.tsv")

def generate_data_test(train_file, val_file, test_file):
    train_ = pd.read_csv(train_file, engine='python', encoding='ISO-8859-1')
    val_ = pd.read_csv(val_file, engine='python', encoding='ISO-8859-1')
    test_ = pd.read_csv(test_file, engine='python', encoding='ISO-8859-1')

    train_test = train_.iloc[:5]
    val_test = val_.iloc[0:5]
    test_test = test_.iloc[0:5]

    train_test.to_csv("data/train_test.csv", index=False)
    val_test.to_csv("data/val_test.csv", index=False)
    test_test.to_csv("data/test_test.csv", index=False)


# generate_data_test("data/train.csv", "data/val.csv", "data/test.csv")


def tokenizer(text):
    return [token.text for token in spacy_en.tokenizer(text)]


# 从本地加载切分好的数据集
def load_split_datasets(train_example, val_example, test_example, train_fields, val_fields, test_fields):
    # 加载examples
    with open(train_example, 'rb')as f:
        train_examples = pickle.load(f)
    with open(val_example, 'rb')as f:
        dev_examples = pickle.load(f)
    with open(test_example, 'rb')as f:
        test_examples = pickle.load(f)

    # 恢复数据集
    train = data.Dataset(examples=train_examples, fields=train_fields)
    dev = data.Dataset(examples=dev_examples, fields=val_fields)
    test = data.Dataset(examples=test_examples, fields=test_fields)
    return train, dev, test


def getData(path, train_file, val_file, test_file, batch_size):
    start_time = time.clock()
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)

    if(os.path.exists(f"data/train_examples.pkl") and
            os.path.exists(f"data/val_examples.pkl") and
            os.path.exists(f"data/train_examples.pkl")):
        train_fields = [('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)]
        val_fields = train_fields
        test_fileds = [('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT)]

        train, val, test = load_split_datasets('data/train_examples.pkl', 'data/val_examples.pkl',
                                               'data/test_examples.pkl',
                                               train_fields, val_fields, test_fileds)


    else:
        train, val = data.TabularDataset.splits(
            path=path, train=train_file, validation=val_file, format='csv', skip_header=True,
            fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)]
        )
        test = data.TabularDataset.splits(
            path=path, test=test_file, format='csv', skip_header=True,
            fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT)]
        )
        test = test[0]

        pickle_train_file = open('data/train_examples.pkl', 'wb')
        pickle.dump(train.examples, pickle_train_file)
        pickle_train_file.close()

        pickle_val_file = open('data/val_examples.pkl', 'wb')
        pickle.dump(val.examples, pickle_val_file)
        pickle_val_file.close()

        pickle_test_file = open('data/test_examples.pkl', 'wb')
        pickle.dump(test.examples, pickle_test_file)
        pickle_test_file.close()

    time1 = time.clock()
    print('步骤1：\t耗时%.4fs\n' % (time1 - start_time))

    if os.path.exists("data/pickle_vocab_file.pkl"):
        pickle_vocab_file = open('data/pickle_vocab_file.pkl', 'rb+')
        TEXT.vocab = pickle.load(pickle_vocab_file)
        pickle_vocab_file.close()
    else:

        TEXT.build_vocab(train, vectors='glove.6B.50d')  # , max_size=30000)
        TEXT.vocab.vectors.unk_init = init.xavier_uniform
        pickle_vocab_file = open('data/pickle_vocab_file.pkl', 'wb')
        pickle.dump(TEXT.vocab, pickle_vocab_file)
        pickle_vocab_file.close()

    time2 = time.clock()
    print('步骤2：\t耗时%.4fs\n' % (time2 - time1))

    train_iter = data.BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.Phrase),
                                     shuffle=False, device=DEVICE)

    val_iter = data.BucketIterator(val, batch_size=batch_size, sort_key=lambda x: len(x.Phrase),
                                   shuffle=False, device=DEVICE)

    test_iter = data.Iterator(dataset=test, batch_size=batch_size, train=False,
                              sort=False, device=DEVICE)

    len_vocab = len(TEXT.vocab)
    time3 = time.clock()
    print('步骤2：\t耗时%.4fs\n' % (time3 - time2))

    return train_iter, val_iter, test_iter, len_vocab, TEXT.vocab


# train_iter, val_iter, test_iter, len_vocab, vocab = getData(".", "data/train.csv", "data/val.csv", "data/test.csv")

# i = 0
# for batch_idx, batch in enumerate(train_iter):
#     data = batch.Phrase
#     label = batch.Sentiment
#     print(data.shape)
#     exit()
#     i += 1
#
# print(i)



