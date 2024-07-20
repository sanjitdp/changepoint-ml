import torch
import torch.nn as nn


class IntegerIndicator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n):
        ctx.save_for_backward(input)
        ctx.n = n
        return torch.where(
            abs(input - n) < 1, torch.exp(1 - 1 / (1 - (input - n) ** 2)), 0
        )

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        n = ctx.n
        return (
            torch.where(
                abs(input - n) < 1,
                grad_output
                * (
                    2
                    * torch.exp(1 / (n**2 - 2 * n * input + input**2 - 1))
                    * (n - input)
                    / (n**2 - 2 * n * input + input**2 - 1) ** 2
                ),
                0,
            ),
            None,
        )


class FeedforwardCUSUMNetwork(nn.Module):
    def __init__(self, timeseries_length):
        super(FeedforwardCUSUMNetwork, self).__init__()

        self.timeseries_length = timeseries_length
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

    def cusum_loss(self, y_pred, y_true, x):
        cumulative_predictions = torch.cumsum(y_pred, dim=-1)

        cumulative_indicators = torch.stack(
            [
                torch.exp(-((cumulative_predictions - n) ** 2))
                for n in range(self.timeseries_length)
            ]
        )

        means = [
            torch.dot(cumulative_indicators[n], x)
            / (1e-5 + torch.sum(cumulative_indicators[n]))
            for n in range(self.timeseries_length)
        ]

        return sum(
            [
                torch.dot(cumulative_indicators[j], torch.square(x - means[j]))
                for j in range(self.timeseries_length)
            ]
        )

    def zero_parameters(self):
        for p in self.parameters():
            p.data.zero_()

    def loss(self, y_pred, y_true, x):
        return self.cusum_loss(y_pred, y_true, x)


if __name__ == "__main__":
    f = FeedforwardCUSUMNetwork(10)

    print(
        f.cusum_loss(
            torch.tensor(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], requires_grad=True
            ),
            torch.tensor(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], requires_grad=True
            ),
            torch.tensor(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], requires_grad=True
            ),
        )
    )
