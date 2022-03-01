import torch
import re
import jsonlines
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from nltk.stem import WordNetLemmatizer

class SNLI(Dataset):

    def __init__(self, filename, word2idx, idx2word, vocab):
        self.lemmatizer =  WordNetLemmatizer()
        self.label_idx_dict = {"contradiction": 0, "neutral": 1, "entailment": 2}
        self.labels, self.premises, self.hypothesises = prepare_data(filename,
                                                                     label_idx_dict=self.label_idx_dict,
                                                                     word2idx=word2idx,
                                                                     max_len=40,
                                                                     lemmatizer=self.lemmatizer)

    def __getitem__(self, idx):
        return [self.labels[idx], self.premises[idx], self.hypothesises[idx]]

    def __len__(self):
        return len(self.premises)

    def collate_fn(self, batch_data):

        labels = [data[0] for data in batch_data]
        premises = [data[1] for data in batch_data]
        hypothesises = [data[2] for data in batch_data]

        return [torch.tensor(labels), torch.tensor(premises), torch.tensor(hypothesises)]


# word to index
def word_2_index(word, word2idx):
    return word2idx[word]


# index to word
def index_2_world(index, idx2word):
    return idx2word[index]


# sentence to indexList
def sen2idx(sentence, word2idx, max_len, lemmatizer):
    idx_sen = []
    for i, word in enumerate(sentence):
        word = word.lower()
        # 进行词形还原
        lemmatizer.lemmatize(word)
        # 处理OOV问题
        if word not in word2idx:
            word = "unk"
        if i >= max_len:
            break  # 防止序列过长进行截断
        idx_sen.append(word2idx[word])

    # PADDING
    if len(sentence) < max_len:
        for index in range(max_len - len(sentence)):
            idx_sen.append(0)

    return idx_sen

# a  = ["I", "am", "a" , "boy"]
# word2idx, idx2word, vocab = load_vocab(r"../glove_data/wordsList.npy")

def tokenizer(text):
    """
    Clean text
    :param text: the string of text
    :return: text string after cleaning
    """
    # acronym
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)

    # spelling correction

    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)


    # punctuation
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r"\\", " \ ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"&", " & ", text)
    text = re.sub(r"\|", " | ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ( ", text)

    # symbol replacement
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\$", " dollar ", text)

    # remove extra space
    text = ' '.join(text.split())

    # tokenize
    token_list = text.split(" ")

    return token_list


def label_2_index(label, label_idx_dict):
    return label_idx_dict[label]


def prepare_data(filename, label_idx_dict, word2idx, max_len, lemmatizer):
    labels = []
    premises = []
    hypothesises = []
    with jsonlines.open(filename, "r") as f:
        i = 0
        for line in f:
            # 若最终标签不确定则跳过
            if line["gold_label"] == '-':
                continue
            labels.append(label_2_index(line["gold_label"], label_idx_dict))
            premises.append(sen2idx(tokenizer(line["sentence1"]), word2idx, max_len, lemmatizer))
            hypothesises.append(sen2idx(tokenizer(line["sentence2"]), word2idx, max_len, lemmatizer))
            i = i + 1

    return labels, premises, hypothesises


# train_data = SNLI(r"../snli_1.0/snli_1.0_train.jsonl")
# train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=4, collate_fn=train_data.collate_fn)
#
# for batch_data in train_dataloader:
#     print(batch_data)
#     exit()

