import math

import numpy as np
import torch
import torch.nn as nn
from torch import functional as F

class CategoryDense(nn.Module):
    def __init__(
            self,
            units,
            activation=None
    ):
        super(CategoryDense, self).__init__()

        self.units = units
        self.activation = activation

    def build(self, input_shape):
        category = input_shape[1]
        input_channel = input_shape[2]
        output_channel = self.units
        kernel_shape = [1, category, input_channel, output_channel]
        bias_shape = [1, category, output_channel]
        self.kernel = nn.Parameter(torch.Tensor(*kernel_shape))
        self.bias = nn.Parameter(torch.Tensor(*bias_shape))
        torch.nn.init.xavier_uniform_(self.kernel)
        torch.nn.init.xavier_uniform_(self.bias)


    def forward(self, inputs, **kwargs):
        self.build(input_shape=inputs.shape)
        inputs = inputs[:, :, :, None]
        outputs = torch.sum(torch.mul(inputs, self.kernel.to(inputs.device)), dim=2)
        outputs = torch.add(outputs, self.bias.to(inputs.device))
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            q_d_model: int = 512,
            k_d_model: int = 512,
            v_d_model: int = 512,
            num_dimensions: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(MultiHeadAttention, self).__init__()
        assert num_dimensions % num_heads == 0, "num_dimensions % num_heads should be zero."
        self.num_dimensions = num_dimensions
        self.d_head = int(num_dimensions / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(num_dimensions)

        self.wq = nn.Linear(q_d_model, num_dimensions)
        self.wk = nn.Linear(k_d_model, num_dimensions)
        self.wv = nn.Linear(v_d_model, num_dimensions)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.dense = nn.Linear(num_dimensions, num_dimensions)

    def forward(self, q, k, v, mask = None):
        batch_size = q.size(0)

        query = self.wq(q).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.wk(k).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.wv(v).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))

        score = (content_score) / self.sqrt_dim

        if mask is not None:
            if mask.dtype == torch.bool:
                score.masked_fill_(mask, float('-inf'))
            else:
                score += mask

        attn = torch.softmax(score, -1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.num_dimensions)

        return self.dense(context), score


class DeepAtt(nn.Module):
    def __init__(self, sequence_length, n_targets):
        super(DeepAtt, self).__init__()
        self.conv_pool_drop_1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=1024, kernel_size=30, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=15, stride=15),
            nn.Dropout(0.2))

        self.bidirectional_rnn = nn.LSTM(input_size=1024, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)

        self.category_encoding = torch.eye(n_targets)[None, :, :]

        self.multi_head_attention = MultiHeadAttention(q_d_model=n_targets, k_d_model=1024, v_d_model=1024,
                                                       num_dimensions=400, num_heads=4)

        self.dropout_2 = nn.Dropout(0.2)

        self.point_wise_dense_1 = nn.Sequential(
            nn.Linear(400, 100),
            nn.ReLU())

        self.point_wise_dense_2 = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid())

    def forward(self, inputs, training=None, mask=None, **kwargs):
        batch_size = inputs.shape[0]

        temp = self.conv_pool_drop_1(inputs)
        temp = temp.transpose(1, 2)

        temp, _ = self.bidirectional_rnn(temp)

        query = torch.tile(self.category_encoding, dims=[batch_size, 1, 1]).to(inputs.device)
        temp, _ = self.multi_head_attention(q=query, k=temp, v=temp)

        temp = self.dropout_2(temp)

        temp = self.point_wise_dense_1(temp)

        output = self.point_wise_dense_2(temp)

        output = output.reshape([-1, 19])
        return output


def criterion():
    return nn.BCELoss()

def get_optimizer(lr):
    return (torch.optim.Adam,{"lr": lr, "weight_decay": 1e-6})

