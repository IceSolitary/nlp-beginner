import torch
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CONLL2003(Dataset):

    def __init__(self, filename, word2idx, idx2word, max_len):
        self.labels_idx_dict = \
            {'O': 8, 'B-ORG': 0, 'B-MISC': 1, 'B-PER': 2, 'I-PER': 3, 'B-LOC': 4, 'I-ORG': 5, 'I-MISC': 6, 'I-LOC': 7,
             'START': 9, "END": 10, "PAD": 11}
        raw_data = self.read_data(filename)
        self.sentences, self.labels_seq = self.prepare_data(raw_data, word2idx, self.labels_idx_dict,
                                                            max_len=max_len)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels_seq[idx]

    def __len__(self):
        return len(self.sentences)

    def collate_fn(self, batch_data):
        sentences = [data[0] for data in batch_data]
        labels_seq = [data[1] for data in batch_data]

        return [torch.tensor(sentences), torch.tensor(labels_seq)]

    # word to index
    def word_2_index(self, word, word2idx):
        return word2idx[word]

    # index to word
    def index_2_world(self, index, idx2word):
        return idx2word[index]

    def label_2_index(self, label, label_idx_dict):
        return label_idx_dict[label]

    # sentence to indexList
    def seq2idx(self, contents, word2idx,labels_idx_dict, max_len):
        # contents.shape[2, seq_length] contents=[[tokens_seq],[labels_seq]]
        token_seq = []
        label_seq = []
        # 句子起始添加start标识
        token_seq.append(word2idx["START_TAG"])
        label_seq.append(labels_idx_dict["START"])
        for i, content in enumerate(zip(contents[0],contents[1])):

            # 处理OOV问题
            word = content[0].lower()
            label = content[1]
            if word not in word2idx:
                word = "unk"
                label = "O"
            if i >= max_len -2:
                break  # 防止序列过长进行截断,因为补了start和end所以减去2
            token_seq.append(word2idx[word])
            label_seq.append(labels_idx_dict[label])
        # PADDING
        if len(token_seq) < max_len - 1:
            for index in range(max_len - len(token_seq) - 1):
                token_seq.append(word2idx["PAD_TAG"])
                label_seq.append(labels_idx_dict["PAD"])
        # 句子结束添加end标识
        token_seq.append(word2idx["END_TAG"])
        label_seq.append(labels_idx_dict["END"])

        if len(token_seq) != max_len:
            print("token: ",len(token_seq))

        return token_seq, label_seq

    def read_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            contents = []
            tokens = []
            labels = []
            for line in f:
                line = line.strip('\n')

                if line == '-DOCSTART- -X- -X- O':
                    pass

                elif line == "":
                    if len(tokens):

                        contents.append([tokens, labels])
                        tokens = []
                        labels = []

                else:
                    content = line.split(" ")
                    tokens.append(content[0])
                    labels.append(content[3])

        return contents

    def prepare_data(self, raw_data, word2idx, labels_idx_dict, max_len):
        token_seqs = []
        label_seqs = []
        for content in raw_data:
            token_seq, label_seq = self.seq2idx(content, word2idx, labels_idx_dict, max_len)
            token_seqs.append(token_seq)
            label_seqs.append(label_seq)

        return token_seqs,label_seqs


