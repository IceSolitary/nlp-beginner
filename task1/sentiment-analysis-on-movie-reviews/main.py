import numpy as np, argparse, time, pickle, random
import os
from dataLoader import getData
from model import model
import matplotlib.pyplot as plt

if __name__ == '__main__':
    feats = 'Bow'
    phrase_vec_train, phrase_vec_test, sentiment_train, sentiment_test, vocab = getData(feats=feats)
    # print(phrase_vec_test.shape, phrase_vec_train.shape)
    # print(sentiment_train.shape)

    dims = len(vocab)

    data_size = phrase_vec_train.shape[0]
    batch_num = 16
    batch_size = data_size//batch_num
    epochs = 20

    model_ = model(dim=dims, batch_size=batch_size, data_length=data_size)
    sentiment_train = model_.to_one_hot(sentiment_train)
    sentiment_test = model_.to_one_hot(sentiment_test)

    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx in range(batch_num):
            sentiment_train_batch = sentiment_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            phrase_vec_train_batch = phrase_vec_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            loss = model_.optimize(phrase_vec_train_batch, sentiment_train_batch, num_iterations=100)
            epoch_loss += loss
        print("Epoch{:d} loss:{:4f}".format(epoch, epoch_loss/batch_num))
        epoch_losses.append(epoch_loss/batch_num)

    accuracy = model_.eval(phrase_vec_test, sentiment_test)
    print("Test Accuracy: {:.4f}".format(accuracy))

    plt.title(f'Feature:{feats}')
    plt.plot(np.arange(len(epoch_losses)), np.array(epoch_losses))
    plt.show()

    # model_.optimize(phrase_vec_train, sentiment_train, num_iterations=1000, print_cost=True)

    # phrase_vec_, vocab_ = getData(file="test.tsv/test.tsv", attr='test',vocabulary=vocab)
    # dims_ = len(vocab_)
    # print(dims_)
    # data_size = phrase_vec_.shape[0]
    # batch_size = phrase_vec_.shape[0]
    #
    # model_.predict(phrase_vec_)
