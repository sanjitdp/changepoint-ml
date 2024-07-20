import torch
import torch.nn as nn


class FeedforwardEarthmoverNetwork(nn.Module):
    def __init__(self, timeseries_length):
        super(FeedforwardEarthmoverNetwork, self).__init__()
        self.fc1 = nn.Linear(timeseries_length, 2 * timeseries_length)
        self.fc2 = nn.Linear(2 * timeseries_length, 4 * timeseries_length)
        self.fc3 = nn.Linear(4 * timeseries_length, timeseries_length)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.zero_parameters()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    def earthmover_loss(self, y_pred, y_true):
        return torch.mean(
            torch.square(torch.cumsum(y_pred, dim=-1) - torch.cumsum(y_true, dim=-1)),
            dim=-1,
        )

    def zero_parameters(self):
        for p in self.parameters():
            p.data.zero_()

    def loss(self, y_pred, y_true, x):
        return self.earthmover_loss(y_pred, y_true, x)
