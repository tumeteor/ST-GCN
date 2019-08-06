from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable as V
import torch
import torch.nn as nn
import numpy as np


class NMFS(nn.Module):
    def __init__(self, num_nodes, num_factors):
        super().__init__()
        self.W = nn.Embedding(num_nodes, num_factors, sparse=True)
        self.H = nn.Embedding(num_nodes, num_factors, sparse=True)

        self.w_bias = nn.Embedding(num_nodes, 1, sparse=True)
        self.h_bias = nn.Embedding(num_nodes, 1, sparse=True)

    def forward(self, nodes):
        pred = self.w_bias(nodes) + self.h_bias(nodes)
        pred += (self.W(nodes) * self.H(nodes)).sum(dim=1, keepdim=True)

        return pred.squeeze()


class NMF(nn.Module):
    def __init__(self, in_nodes, out_nodes, num_factors):
        super().__init__()
        self.W = nn.Embedding(in_nodes, num_factors)
        self.H = nn.Embedding(out_nodes, num_factors)

        self.W.weight.data.uniform_(-.01, .01)
        self.H.weight.data.uniform_(-.01, .01)

        self.w_bias = nn.Embedding(in_nodes, 1)
        self.h_bias = nn.Embedding(out_nodes, 1)

        self.w_bias.weight.data.uniform_(-0.1, .01)
        self.h_bias.weight.data.uniform_(-0.1, .01)

    def forward(self, nodes):
        in_nodes, out_nodes = nodes[:, 0], nodes[:, 1]
        pred = (self.W(in_nodes) * self.H(out_nodes)).sum(1)
        pred += self.w_bias(in_nodes).squeeze() + self.h_bias(out_nodes).squeeze()
        return pred


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


def train_nmf_with_sparse_matrix(dataset, num_factors=10, lr=1e-4, num_epochs=1000):
    nmf = NMFS(dataset.shape[0], num_factors)
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


def train_nmf_with_dataframe(dataset, num_nodes, num_factors=10, lr=1e-4, num_epochs=1000):
    nmf = NMF(num_nodes, num_nodes, num_factors)
    optimizer = torch.optim.SGD(nmf.parameters(), lr=lr)
    train_loader = get_loader(dataset)
    loss_func = torch.nn.MSELoss()
    training_loss = []
    for epoch in range(num_epochs):
        for i, train_data in enumerate(train_loader):
            # get the inputs
            inputs = train_data.long()
            actual_out = V(train_data[:, 2].float())
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = nmf.forward(inputs)

            loss = loss_func(outputs, actual_out)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

            # input()
            if i % (len(train_loader)) == (len(train_loader) - 1):  # print every 2000 mini-batches
                training_loss.append((running_loss / len(train_loader)))
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / len(train_loader)}")
                # epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


