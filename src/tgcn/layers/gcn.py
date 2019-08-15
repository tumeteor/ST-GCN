from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import math
import torch


class GraphConvolution(nn.Module):

    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """
    Args:
        in_feats (int): number of input features on each node
        hidden_size (int): number of hidden sizes
        out_feats (int): number of output features
    """
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(in_feats, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, out_feats)
        self.linear = nn.Linear(hidden_size, out_feats)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, training=self.training)
        return self.linear(x)

