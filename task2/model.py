import torch
from torch import nn
import torch.nn.functional as F
import time


class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 50)
        self.lstm = nn.LSTM(50, 128, 3, batch_first=True)  # ,bidirectional=True)
        self.linear = nn.Linear(128, 5)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, length):
        batch_size, seq_num = x.shape
        vec = self.embedding(x)
        vec = self.dropout(vec)
        out, (hn, cn) = self.lstm(vec)
        out = self.linear(out)

        out_ = out[range(len(length)), length - 1]

        return out_


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout=0.5, num_filters=64, num_classes=5):
        super(TextCNN, self).__init__()
        self.filter_sizes = [2, 3, 4]
        self.num_filters = num_filters

        self.embedding = nn.Embedding(vocab_size, 50)
        self.CNNs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (filter_size, embedding_size)) for filter_size in self.filter_sizes]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(num_filters*len(self.filter_sizes), num_classes)

    def conv_and_pool(self, inputs, conv):
        inputs = inputs.unsqueeze(1)
        features = F.relu(conv(inputs)).squeeze(-1)
        max_pooled_feats = F.max_pool1d(features, features.size(-1)).squeeze(-1)

        return max_pooled_feats

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        cnn_features = []
        for cnn in self.CNNs:
            cnn_features.append(self.conv_and_pool(embedded, cnn))
        cnn_features = torch.cat(cnn_features, dim=-1)
        cnn_features = self.dropout(cnn_features)
        pred = self.fc(cnn_features)

        return pred


def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
