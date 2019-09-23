from torch import nn
import torch.nn.functional as F

import torch
import math

from torch.nn import Parameter

torch.manual_seed(0)


class GCLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GCLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if bias:
            self.bias = Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.gcn_weight = Parameter(torch.FloatTensor(input_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x, hx, cx, adj):
        support = torch.mm(x.float(), self.gcn_weight)
        x = torch.spmm(adj, support)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        if self.bias is not None:
            x = x + self.bias

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
