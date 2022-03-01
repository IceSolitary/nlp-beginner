import torch
import torch.nn as nn
import math
from model.layers import InputVariationalDropout, EmbeddingLayer, EncodingLayer


class SLModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 batch_size,
                 label_size,
                 tag_to_ix,
                 device,
                 embedding,
                 dropout=0.5):

        super(SLModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.label_size = label_size
        self.dropout = dropout
        self.tag_to_ix = tag_to_ix
        self.device = device

        # Word Embedding
        self.dropoutLayer = InputVariationalDropout(p=dropout)
        self.embeddingLayer = EmbeddingLayer(vocab_size, embedding_size, embedding=embedding)

        # Word Encoding (BiLSTM)
        self.encodingLayer = EncodingLayer(embedding_size, hidden_size)

        # Fully Connected Layer 全连接层提取信息，输出为(batch_size,seq_length,label_size) 为发射矩阵，表示token到label的分数
        self.fc = nn.Linear(2 * hidden_size, label_size, bias=True)

        # CRF Layer
        # 转移矩阵初始化，调整为可训练参数，转移矩阵为表示标签到标签的转移概率
        self.transitions = nn.Parameter(
            torch.randn(self.label_size, self.label_size)
        ).to(device)
        # 所有标签到起始标签的转移概率为0
        self.transitions.data[tag_to_ix["START"], :] = -10000
        # 结束标签到其它所有标签的转移概率为0
        self.transitions.data[:, tag_to_ix["END"]] = -10000
        # PAD标签对其他所有标签转移概率为0
        self.transitions.data[:, tag_to_ix["PAD"]] = -10000
        # END到PAD标签转移概率不为0
        self.transitions.data[tag_to_ix["END"], tag_to_ix["PAD"]] = 0
        # PAD到PAD标签转移概率不为0
        self.transitions.data[tag_to_ix["PAD"], tag_to_ix["PAD"]] = 0

        self.apply(_init_esim_weights)

    # use DP to calculate partition function
    # 通过best_path_sum  shape(1,label_size) 记录某时刻t对应的所有标签的路径的指数和的对数
    def partition(self, emission, mask):
        """
        配分函数的计算
        :param emission: 发射矩阵 [batch_size, seq_length, label_size]
        :param mask: 掩码 [batch_size, seq_length]
        :return: path_sum 配分函数的值的对数
        """
        batch_size, seq_length, label_size = emission.size()
        init_path_sum = torch.full((batch_size, label_size), -10000.).to(self.device)  # .to(self.device)  # t时刻节点i的路径和
        init_path_sum[:, self.tag_to_ix["START"]] = 0
        forward_path_sum = init_path_sum.clone()  # 暂存 shape [batch_size, label_size]

        for t in range(1, seq_length):
            # 时间步t上，应用动态规划
            temp_t = []
            if mask is not None:
                t_pad = mask[:, t]
                for label_i in range(label_size):
                    emit_score = emission[:, t, label_i].unsqueeze(-1).expand(batch_size, label_size)
                    trans_score = self.transitions[label_i, :].unsqueeze(0).expand(batch_size, label_size)

                    next_path_sum = forward_path_sum + trans_score * t_pad.unsqueeze(-1) + \
                                    emit_score * t_pad.unsqueeze(-1)
                    temp_t.append(log_sum_exp(next_path_sum).view(batch_size, 1))
                forward_path_sum = torch.cat(temp_t, dim=-1)
            else:
                for label_i in range(label_size):
                    emit_score = emission[:, t, label_i].unsqueeze(-1).expand(batch_size, label_size)
                    trans_score = self.transitions[label_i, :].unsqueeze(0).expand(batch_size, label_size)

                    next_path_sum = forward_path_sum + trans_score + emit_score
                    temp_t.append(log_sum_exp(next_path_sum).view(batch_size, 1))
                forward_path_sum = torch.cat(temp_t, dim=-1)
        total_path_sum = log_sum_exp(forward_path_sum)

        return total_path_sum

    def get_path_score(self, token_seq, label_seq, mask, emission):
        """
        获得单条路径得分
        :param token_seq: 词元序列，[batch_size,seq_length]
        :param label_seq: 标签序列，[batch_size,seq_length]
        :param mask: 掩码， [batch_size,seq_length]
        :param emission: 发射矩阵， [batch_size,seq_length,label_size]
        :return: score: 单条路径得分 [batch_size]
        """
        batch_size, seq_length = token_seq.size()
        score = torch.zeros(batch_size).to(self.device)
        for t in range(0, seq_length):
            if mask is not None:
                t_pad = mask[:, t]
                score = score + (emission[:, t, :].gather(1, label_seq[:, t].unsqueeze(-1))).squeeze(-1) * t_pad
                if t != seq_length - 1:
                    score = score + self.transitions[label_seq[:, t + 1], label_seq[:, t]] * t_pad
            else:
                score = score + emission[:, t, :].gather(1, label_seq[:, t].unsqueeze(-1)).squeeze(-1)
                if t != seq_length - 1:
                    score = score + self.transitions[label_seq[:, t + 1], label_seq[:, t]]

        return score

    def forward(self, token_seq, label_seq, mask):
        # 编码为词向量

        embedded_seq = self.embeddingLayer(token_seq)
        if self.dropout:
            embedded_seq = self.dropoutLayer(embedded_seq)
        # 词向量编码
        encoded_seq = self.encodingLayer(embedded_seq)
        # 全连接层形成发射矩阵
        emission = self.fc(encoded_seq)
        # 计算单条路径分数
        path_score = self.get_path_score(token_seq, label_seq, mask, emission)

        # 计算配分函数的值（log）
        partition_score = self.partition(emission, mask)

        loss = -(path_score - partition_score)

        return loss

    def predict(self, token_seq, mask):
        embedded = self.embeddingLayer(token_seq)
        encoded = self.encodingLayer(embedded)
        emission = self.fc(encoded)
        max_score, max_seq = self.viterbi_decode(emission, mask)
        return max_score, max_seq

    def viterbi_decode(self, emission, mask):
        """
        维特比算法用于解码，预测标注序列
        :param mask: 掩码 [batch_size,seq_length]
        :param emission: 发射矩阵[batch_size, seq_length, label_size]
        :return: best_path: 最优标注序列 [batch_size,seq_length]
        """

        batch_size, seq_length, label_size = emission.size()
        # 输入中在句首句尾补足了“START”和“END”，因此句首第一个单词一定为“START”
        # 因此初始化第一个状态从发射矩阵的下标1开始，加上START到各个标签的转移分数
        init_score = self.transitions[:, self.tag_to_ix["START"]].unsqueeze(0).expand(batch_size, label_size) + emission[:, 1]
        init_score.to(self.device)
        init_label = torch.cat([
            torch.tensor([self.tag_to_ix["START"]]).unsqueeze(0).unsqueeze(1).expand(batch_size, label_size, 1),
            torch.arange(0, label_size).unsqueeze(0).expand(batch_size, label_size).unsqueeze(-1)
        ], dim=-1)
        init_label.to(self.device)
        label_seq = init_label.clone().to(self.device)
        label_score = init_score.clone()

        for t in range(2, seq_length-1):

            if mask is not None:
                t_pad = mask[:, t].to(self.device)
                forward_tag = label_seq[:, :, -1].to(self.device)
                # 将当前时间步的emission加到transition的列上，表示在当前时间步发射到相应的label要加的分数，两数之和为当前时间步的总分
                total_score = self.transitions.unsqueeze(0).expand(batch_size, label_size, label_size) \
                              + emission[:, t, :].unsqueeze(-1)
                # 此操作会把transitions矩阵中的元素以forward_tag中的列下标，按列取出
                forward_to_next_trans = total_score.gather(2,forward_tag.reshape(batch_size, 1, label_size).
                                                           expand(batch_size, label_size, label_size)).to(self.device)
                forward_to_next_trans = forward_to_next_trans.transpose(1,
                                                                        2)  # shape[batch_size, label_size, label_size]
                # 求出下一步的最大分数和路径节点下标
                next_max_score, next_max_idx = torch.max(forward_to_next_trans, dim=-1)  # shape[batch_size, label_size]
                label_score += t_pad.unsqueeze(-1) * next_max_score
                next_max_idx = next_max_idx * t_pad.unsqueeze(-1) + (1 - t_pad).unsqueeze(-1) * torch.full((batch_size, label_size), 1).to(self.device) * \
                               self.tag_to_ix["PAD"]
                next_max_idx = next_max_idx.unsqueeze(-1).to(self.device)

                label_seq = torch.cat([label_seq, next_max_idx], dim=-1)

            else:

                forward_tag = label_seq[:, :, -1]
                # 将当前时间步的emission加到transition的列上，表示在当前时间步发射到相应的label要加的分数，两数之和为当前时间步的总分
                total_score = self.transitions.unsqueeze(0).expand(batch_size, label_size, label_size) \
                              + emission[:, t, :].unsqueeze(-1)
                # 此操作会把transitions矩阵中的元素以forward_tag中的列下标，按列取出
                forward_to_next_trans = total_score.gather(2, forward_tag.reshape(batch_size, 1, label_size).
                                                           expand(batch_size, label_size, label_size))
                forward_to_next_trans = forward_to_next_trans.transpose(1, 2)
                # shape[batch_size, label_size, label_size]
                # 求出下一步的最大分数和路径节点下标
                next_max_score, next_max_idx = torch.max(forward_to_next_trans, dim=-1)  # shape[batch_size, label_size]
                label_score += next_max_score
                next_max_idx = next_max_idx.unsqueeze(-1)
                label_seq = torch.cat([label_seq, next_max_idx], dim=-1)

        max_score, max_idx = torch.max(label_score, dim=-1)
        max_seq = label_seq.gather(1, max_idx.reshape(batch_size, 1, 1).expand(batch_size, 1, seq_length-1)).squeeze(1)

        max_seq = torch.cat([
            max_seq , torch.full((batch_size, 1), 1).to(self.device) * self.tag_to_ix["END"]
        ], dim=-1)

        return max_score, max_seq


def log_sum_exp(vec):
    """
    :param vec: [batch_size, label_size]
    :return: vec: [batch_size, label_size]
    """
    max_score, max_idx = torch.max(vec, dim=1)
    max_score_broadcast = max_score.unsqueeze(-1)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


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
        module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0
