import torch
import math
import torch.nn as nn
from torch.nn import Parameter as P
from torch.autograd import Variable as V


class LayerNorm(nn.Module):

    def __init__(self, input_size, learnable=True, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = torch.Tensor(1, input_size).fill_(0)
        self.beta = torch.Tensor(1, input_size).fill_(0)
        self.epsilon = epsilon
        # Wrap as parameters if necessary
        if learnable:
            W = P
        else:
            W = V
        self.alpha = W(self.alpha)
        self.beta = W(self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - torch.mean(x, 1).expand_as(x)) / torch.sqrt(torch.var(x, 1).expand_as(x) + self.epsilon)
        if self.learnable:
            x = self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)

