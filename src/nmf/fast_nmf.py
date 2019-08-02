from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np


class NMF(nn.Module):

    def __init__(self, num_nodes, num_factors):
        super().__init__()
        self.W = nn.Embedding(num_nodes, num_factors, sparse=True)
        self.H = nn.Embedding(num_nodes, num_factors, sparse=True)

        # self.W.weight.data.uniform_(-.01, .01)
        # self.H.weight.data.uniform_(-.01, .01)

        self.w_bias = nn.Embedding(num_nodes, 1, sparse=True)
        self.h_bias = nn.Embedding(num_nodes, 1, sparse=True)

        # self.w_bias.weight.data.uniform_(-0.1, .01)
        # self.h_bias.weight.data.uniform_(-0.1, .01)

    def forward(self, nodes):
        pred = self.w_bias(nodes) + self.h_bias(nodes)
        pred += (self.W(nodes) * self.H(nodes)).sum(dim=1, keepdim=True)
        return pred.squeeze()


class SpeedDataset(Dataset):
    """Create custom class for pytorch data set"""

    def __init__(self, df):
        """Initialization of data frame"""
        self.data = df

    def __len__(self):
        """find total number of samples"""
        return self.data.shape[0]

    def __getitem__(self, index):
        """Generates one sample of data"""
        # get data sample from the data set
        return np.array(self.data.loc[index, :])


def get_loader(df):
    return DataLoader(SpeedDataset(df), batch_size=32, num_workers=1)


def train_nmf(dataset, num_factors=10, lr=1e-4, num_epochs=1000):
    nmf = NMF(dataset.shape[0], num_factors)
    optimizer = torch.optim.SparseAdam(nmf.parameters(), lr=lr)
    rows, cols = dataset.nonzero()
    p = np.random.permutation(len(rows))
    rows, cols = rows[p], cols[p]
    loss_func = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        batch_loss = 0
        optimizer.zero_grad()

        for row, col in zip(*(rows, cols)):
            # Turn data into tensors
            speed = torch.FloatTensor([dataset[row, col]])
            row = torch.LongTensor([row])

            # Predict and calculate loss
            prediction = nmf(row)
            loss = loss_func(prediction, speed)
            batch_loss += loss.item()
            # Backpropagate
            loss.backward()

            # Update the parameters
            optimizer.step()
        print(f"loss at step {epoch} is: {batch_loss}")




