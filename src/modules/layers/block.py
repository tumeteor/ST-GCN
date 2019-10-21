import torch

import torch.nn.functional as F
import torch.nn as nn
import math


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.

    Args:
            in_channels: Number of input features at each node in each time
               step.
            out_channels: Desired number of output channels at each node in
               each time step.
            kernel_size: Size of the 1D temporal kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        Args:
           X: Input data of shape (batch_size, num_nodes, num_timesteps,
               num_features=in_channels)

        Returns:
           Output data of shape (batch_size, num_nodes,
               num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    Args:
            in_channels: Number of input features at each node in each time
                step.
            spatial_channels: Number of output channels of the graph
                convolutional, spatial sub-block.
            out_channels: Desired number of output features at each node in
                each time step.
            num_nodes: Number of nodes in the graph.
    """
    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        Args:
            X: Input data of shape (batch_size, num_nodes, num_timesteps,
              num_features=in_channels).
            A_hat: Normalized adjacency matrix.
              :return: Output data of shape (batch_size, num_nodes,
              num_timesteps_out, num_features=out_channels).

        Returns:

        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat.to_dense(), t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
