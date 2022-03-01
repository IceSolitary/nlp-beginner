from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import gensim


glove_input_file = 'glove.6B/glove.6B.50d.txt'
word2vec_output_file = 'glove.6B/glove.6B.50d.word2vec.txt'

# glove2word2vec(glove_input_file,word2vec_output_file)
# glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# glove_model.save("glove.6B.50d.model")

model = gensim.models.KeyedVectors.load('glove.6B.50d.model')
word = u'me'
if word in model:
    print(model[word])

# def getData(file="train.tsv/train.tsv", attr='train_and_test', vocabulary={}):
#     dataset = pd.read_csv(file, sep='\t', engine='python', encoding='ISO-8859-1')
#
#     # TODO: Remove stopwords
#
#     phrase_ = dataset['Phrase'].values
#     phrase = np.array(phrase_)
#     if attr == 'train_and_test' or attr == 'train':
#         sentiment_ = dataset['Sentiment'].values
#         sentiment = np.array(sentiment_)
#
#     glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
#     # phrase_vec, dims = bow.get_Bow(phrase)
#     # phrase_vec, vocab = bow.sklearn_get_Bow(phrase)
#     #
#     # phrase_vec_ = preprocessing.normalize(phrase_vec, axis=1)
#
#     if attr == 'train_and_test':
#         phrase_vec_train, phrase_vec_test, sentiment_train, sentiment_test = train_test_split(phrase_vec, sentiment, test_size=0.2, random_state=5)
#         return phrase_vec_train , phrase_vec_test, sentiment_train, sentiment_test, vocab
#     elif attr == 'train':
#         return phrase_vec_, sentiment, vocab
#     else:
#         return phrase_vec_, vocab