import torch
import math
import copy

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F


class GCN_LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers, adj_mat):
        super().__init__()

        self._cells = []

        for i in range(num_layers):
            cell = GCN_LSTMCell(input_size, hidden_size, adj_mat)
            # NOTE hidden state becomes input to the next cell
            input_size = hidden_size
            self._cells.append(cell)
            # Hook to register submodule
            setattr(self, "cell{}".format(i), cell)

    def forward(self, input):
        # NOTE (seq_len, batch, edges, features)
        batch_size = input.size(1)
        edge_cnt = input.size(2)
        c_states = []
        h_states = []
        outputs = []

        for step, x in enumerate(input):
            for cell_idx, cell in enumerate(self._cells):
                if step == 0:
                    h, c = self._cells[cell_idx].init_hidden(
                        batch_size, edge_cnt, input.device
                    )
                    h_states.append(h)
                    c_states.append(c)

                # NOTE c and h are coming from the previous time stamp, but we iterate over cells
                h, c = cell(x, h_states[cell_idx], c_states[cell_idx])
                h_states[cell_idx] = h
                c_states[cell_idx] = c
                # NOTE hidden state of previous LSTM is passed as input to the next one
                x = h

            outputs.append(h)

        # NOTE Concat along the channels
        return torch.stack(outputs, dim=1)


class GCN_LSTMCell(Module):
    def __init__(self, input_size, hidden_size, adj_mat, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.conv_xf = GraphConv(input_size, hidden_size, adj_mat)
        self.conv_hf = GraphConv(hidden_size, hidden_size, adj_mat, bias=False)

        self.conv_xi = copy.deepcopy(self.conv_xf)
        self.conv_hi = copy.deepcopy(self.conv_hf)

        self.conv_xo = copy.deepcopy(self.conv_xf)
        self.conv_ho = copy.deepcopy(self.conv_hf)

        self.conv_xc = copy.deepcopy(self.conv_xf)
        self.conv_hc = copy.deepcopy(self.conv_hf)

    def init_hidden(self, batch_size, edge_cnt, device):
        h = torch.zeros(batch_size, edge_cnt, self.hidden_size, device=device)
        c = torch.zeros(batch_size, edge_cnt, self.hidden_size, device=device)
        return (h, c)

    def forward(self, x, h, c):
        # Normalized shape for LayerNorm
        normalized_shape = list(h.shape[-2:])

        def LR(input):
            return F.layer_norm(input, normalized_shape)

        i = torch.sigmoid(LR(self.conv_xi(x) + self.conv_hi(h)))
        f = torch.sigmoid(LR(self.conv_xf(x) + self.conv_hf(h)))
        o = torch.sigmoid(LR(self.conv_xo(x) + self.conv_ho(h)))
        c = f * c + i * torch.tanh(LR(self.conv_xc(x) + self.conv_hc(h)))
        h = o * torch.tanh(c)

        return (h, c)


class GraphConv(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj_mat, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_mat = adj_mat
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.adj_mat, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
