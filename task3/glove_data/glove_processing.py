import numpy as np
import os
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

embeddings_dict = {}

print(os.getcwd())

def get_wordsList_and_wordVectors(filepath):
    # with open("glove.840B.300d.txt", 'r', encoding="utf-8") as f:
    with open(filepath, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                if len(values[1:]) != 300:
                    continue
                embeddings_dict[word] = vector

            except ValueError:
                print(i, " ", values)

    np.save('wordsList', np.array(list(embeddings_dict.keys())))
    np.save('wordVectors', np.array(list(embeddings_dict.values()), dtype='float32'))

get_wordsList_and_wordVectors("glove.6B.300d.txt")





