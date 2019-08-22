from torch import optim

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
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
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
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
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat.to_dense(), t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(pl.LightningModule):
    """
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes=6163, num_features=29, num_timesteps_input=9,
                 num_timesteps_output=1, adj=None, datasets=None, targets=None, mask=None):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear(num_timesteps_input * 64,
                               num_timesteps_output)

        self.datasets = datasets
        self.targets = targets
        self.mask = mask

        self.adj = adj

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        print(f"Out3 shape: {out3.shape}")
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def training_step(self, batch, batch_nb):
        # (batch_size, num_nodes, num_input_time_steps, num_features)
        # [batch_size, seq_length, input_dim(features)] 1, 6163, 9, 27
        x, y = batch
        x = x.permute(0, 1, 3, 2).float()

        y = y.squeeze(dim=0)
        y_hat = self.forward(A_hat=self.adj, X=x).squeeze(dim=0)

        _mask = self.mask['train'][batch_nb, :, :]
        y_hat = y_hat.masked_select(_mask)
        y = y.masked_select(_mask)

        return {'loss': F.mse_loss(y_hat, y.float())}

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(TensorDataset(self.datasets['train'], self.targets['train']), batch_size=1, shuffle=False)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(TensorDataset(self.datasets['valid'], self.targets['valid']), batch_size=1, shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(TensorDataset(self.datasets['test'], self.targets['test']), batch_size=1, shuffle=False)

    def validation_step(self, batch, batch_nb):
        # batch shape: torch.Size([1, 6163, 26, 10])
        # [batch_size, seq_length, input_dim(features)] 1, 6163, 9, 27
        x, y = batch
        x = x.permute(0, 1, 3, 2).float()

        y = y.squeeze(dim=0)

        y_hat = self.forward(A_hat=self.adj, X=x).squeeze(dim=0)
        _mask = self.mask['valid'][batch_nb, :, :]

        y_hat = y_hat.masked_select(_mask)
        y = y.masked_select(_mask)

        print(f"y_hat: {y_hat}")
        mae = (y_hat - y.float()).abs().sum() / _mask.sum()
        return {'val_mae': mae}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_mae'] for x in outputs]).mean()
        return {'avg_val_mae': avg_loss}

