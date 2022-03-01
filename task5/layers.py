import torch
import torch.nn as nn

class InputVariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, [Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) to a
    3D tensor.
    This module accepts a 3D tensor of shape `(batch_size, num_timesteps, embedding_dim)`
    and samples a single dropout mask of shape `(batch_size, embedding_dim)` and applies
    it to every time step.
    """

    def forward(self, input_tensor):

        """
        Apply dropout to input tensor.
        # Parameters
        input_tensor : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)`
        # Returns
        output : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor

class EmbeddingLayer(nn.Module):

    def __init__(self,
                 vocab_size,
                 embeddding_size,
                 dropout=0.5,
                 padding_idx=0,
                 embedding=None):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embeddding_size = embeddding_size
        self.embedding = nn.Embedding(vocab_size, embeddding_size, _weight=embedding, padding_idx=padding_idx)
        self.dropout = dropout
        self.dropout = InputVariationalDropout(dropout)

    def init_weight(self, vectors):
        """

        :param vectors:  预训练的词向量
        :return:
        """
        self.embedding.weight.data.copy_(vectors)
        # self.embedding.requires_grad = False

    def forward(self, input):
        embedding = self.embedding(input)
        embedding_dropout = self.dropout(embedding)

        return embedding_dropout