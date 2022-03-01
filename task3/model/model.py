import torch
import torch.nn as nn

from model.layers import InputVariationalDropout, EmbeddingLayer, EncodingLayer, AttentionLayer, PoolingLayer


class ESIM(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_class = 3,
                 dropout=0.5,
                 padding_idx=0,
                 embedding=None,
                 bidirectional=True,
                 device="cpu"):
        super(ESIM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device

        # Input Encoding
        # 输入的dropout层
        self.dropoutLayer = InputVariationalDropout(p=dropout)
        # embedding层
        self.embeddingLayer = EmbeddingLayer(vocab_size, embedding_size, dropout=dropout,embedding=embedding)
        # encoding层
        self.encodingLayer = EncodingLayer(embedding_size, hidden_size, bidirectional=True)

        # Local Inference Modeling
        # 注意力层
        self.AttentionLayer = AttentionLayer()

        # Inference Composition
        # 前馈网络层
        self.fnn = nn.Sequential(nn.Linear(4*2*hidden_size, hidden_size),  # 双向lstm输出为2*hidden_size
                                 nn.ReLU()) 
        # composition层
        self.compositionLayer = EncodingLayer(hidden_size, hidden_size, dropout=dropout, bidirectional=True)
        # 汇聚层
        self.poolingLayer = PoolingLayer()
        # MLP分类
        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(8*hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_class)
        )

        self.apply(_init_esim_weights)

    def forward(self, premises,hypothesises):

        # print(premises)
        embedded_p = self.embeddingLayer(premises)
        # print("embedded_p", embedded_p)
        embedded_h = self.embeddingLayer(hypothesises)
        if self.dropout:
            embedded_p = self.dropoutLayer(embedded_p)
            embedded_h = self.dropoutLayer(embedded_h)
        # print("embedded_p", embedded_p)

        encoded_p = self.encodingLayer(embedded_p)
        encoded_h = self.encodingLayer(embedded_h)

        # print("encoded_p", encoded_p.shape)

        attention_p, attention_h = self.AttentionLayer(encoded_p, encoded_h)

        # print("attention_p", attention_p)

        enhancement_p = torch.cat([encoded_p, attention_p, encoded_p - attention_p, encoded_p * attention_p], dim=-1)
        enhancement_h = torch.cat([encoded_h, attention_h, encoded_h - attention_h, encoded_h * attention_h], dim=-1)

        fnn_p = self.fnn(enhancement_p)
        fnn_h = self.fnn(enhancement_h)

        if self.dropout:
            fnn_p = self.dropoutLayer(fnn_p)
            fnn_h = self.dropoutLayer(fnn_h)

        v_p = self.compositionLayer(fnn_p)
        v_h = self.compositionLayer(fnn_h)

        avg_v_p, max_v_p = self.poolingLayer(v_p)
        avg_v_h, max_v_h = self.poolingLayer(v_h)

        v = torch.cat([avg_v_p, max_v_p, avg_v_h, max_v_h], dim=-1)

        logits = self.mlp(v)

        logits_ = nn.functional.softmax(logits, dim=-1)


        class_idx = torch.max(logits_, dim=-1)[1]

        return logits_, class_idx

# init code from https://github.com/coetaur0/ESIM/blob/master/esim/model.py
def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0





