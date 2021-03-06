from torch import nn
import torch.nn.functional as F

import torch
import math

from torch.nn import Parameter

torch.manual_seed(0)


class GCLSTMCell(nn.Module):
    """
    Graph convolution LSTM cell.
     Args:
            input_size (int): the input size
            hidden_size (int): the size of the hidden states
            bias (bool): whether to add bias for regularization
            dropout (double): the dropout rate
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.5):
        super(GCLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if bias:
            self.bias = Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.x2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.gcn_weight = Parameter(torch.FloatTensor(input_size, hidden_size))
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x, hx, cx, adj):
        """
        Args:
            x: Input data of shape (batch_size, num_nodes, num_timesteps,
                    num_features=in_channels).
            hx: hidden state vector
            cx: cell state vector
            adj: Normalized adjacency matrix.
                    :return: Output data of shape (batch_size, num_nodes,
                    num_timesteps_out, num_features=out_channels).

        Returns:

        """
        support = torch.mm(x.float(), self.gcn_weight)
        x = torch.spmm(adj, support)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        if self.bias is not None:
            x = x + self.bias
        # apply batch-norm over all nodes (in the batch)
        x = self.batch_norm(x)
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, torch.tanh(cy))

        return hy, cy
