from tqdm import tqdm

from src.models.tgcn.layers.gcn import GCN
from src.models.tgcn.layers.lstm import LSTMs
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--gcn_in', type=int, default=12)
parser.add_argument('--gcn_hid', type=int, default=20)
parser.add_argument('--gcn_out', type=int, default=20)
parser.add_argument('--lstm_hid', type=int, default=32)
parser.add_argument('--lstm_layers', type=int, default=2)
parser.add_argument('--lstm_drop', type=int, default=0)
parser.add_argument('--a_in', type=int, default=10)
parser.add_argument('--output_pred', type=int, default=1)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)


args = parser.parse_args()
net = GCN(args.gcn_in, args.gcn_hid, args.gcn_out)
model = LSTMs(args.gcn_out, args.lstm_hid, args.output_pred, args.lstm_layers, args.lstm_drop)

parameters = list(net.parameters()) + list(model.parameters())
loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(parameters, lr=args.lr)

X_train = []
Adj_train = []
y_train = []

for epoch in tqdm(range(args.epoch)):
    net.train()
    model.train()

    cell = net(X_train, Adj_train)
    y_pred = model(cell)

    loss = loss_fn(y_pred, y_train)
    if epoch % 100 == 0:
        print("Epoch ", epoch, "MSE: ", loss.item())

    # Zero out gradient, else they will accumulate between epochs
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()




