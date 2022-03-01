import torch
import torch.nn as nn
from layers import InputVariationalDropout, EmbeddingLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Seq_Generation_model(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 dropout=0.5,
                 embedding=None,
                 bidirectional=True,
                 device="cpu"):
        super(Seq_Generation_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device
        self.start_id = 0
        self.end_id = 1
        _x = nn.init.xavier_normal_(torch.Tensor(self.vocab_size, self.embedding_size))

        # Embedding
        # self.dropoutLayer = InputVariationalDropout(p=dropout)
        self.embeddingLayer = EmbeddingLayer(vocab_size, embedding_size, dropout=self.dropout, embedding=_x)
        self.encodingLayer = nn.LSTM(input_size=self.embedding_size,
                                    hidden_size=self.hidden_size,
                                    batch_first=True,
                                    dropout=0.5)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input, is_test=False):
        """
        :param input: poem_vector, size:[batch_size, seq_length]
        :return:
        """

        embed_input = self.embeddingLayer(input)
        encoded_input, _ = self.encodingLayer(embed_input)
        output = self.fc(encoded_input)

        if is_test:
            output = output[:, -1, :]
            output = output.topk(1,dim=-1)
            output = output[1][0]

            return output

        return output

    def generate_random_poem(self, max_len, num_sentence, random=False):
        if random:
            initialize = torch.randn
        else:
            initialize = torch.zeros
        hn = initialize((1, 1, self.hidden_size)).cuda()
        cn = initialize((1, 1, self.hidden_size)).cuda()
        x = torch.LongTensor([self.start_id]).cuda()
        poem = list()

        while (len(poem) != num_sentence):
            word = x
            sentence = list()
            for j in range(max_len):
                word = torch.LongTensor([word]).cuda()
                word = self.embeddingLayer(word).view(1, 1, -1)
                output, (hn, cn) = self.encodingLayer(word, (hn, cn))
                output = self.fc(output)
                word = output.topk(1)[1][0].item()

                if word == self.end_id:
                    x = torch.LongTensor([self.start_id]).cuda()
                    break
                sentence.append(self.num_to_word[word])
                if self.word_to_num['。'] == word:
                    break
            else:
                x = self.word_to_num['。']
            if sentence:
                poem.append(sentence)

        return poem















