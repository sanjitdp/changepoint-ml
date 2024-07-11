import numpy as np
import torch
from models.feedforward_earthmover_network import FeedforwardEarthmoverNetwork
import matplotlib.pyplot as plt

TIMESERIES_LENGTH = 1024
CHANGEPOINT_PROB = 0.01
POSSIBLE_MEANS = list(range(-3, 4))
GAUSSIAN_VARIANCE = 1
LOAD_STATE = "src/trained_models/feedforward_earthmover_network_weights_1024.pth"

model = FeedforwardEarthmoverNetwork(TIMESERIES_LENGTH)
model.load_state_dict(torch.load(LOAD_STATE))
model.eval()

with torch.no_grad():
    x_test = np.zeros((1, TIMESERIES_LENGTH))
    y_test = np.zeros((1, TIMESERIES_LENGTH))

    curr_mean = 0
    for j in range(TIMESERIES_LENGTH):
        coin = np.random.choice([0, 1], p=[1 - CHANGEPOINT_PROB, CHANGEPOINT_PROB])
        if coin:
            new_mean = np.random.choice(POSSIBLE_MEANS)
            if new_mean != curr_mean:
                y_test[0][j] = 1
                curr_mean = new_mean

        x_test[0][j] = np.random.normal(curr_mean, GAUSSIAN_VARIANCE)

    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    y_pred = model(x_test_tensor)
    print("Predicted changepoints:", y_pred)

    test_loss = model.earthmover_loss(y_pred, y_test_tensor).mean().item()
    print(f"Mean testing loss: {test_loss}")

    plt.title("Gaussian time series with mean change")
    plt.scatter(range(1, TIMESERIES_LENGTH + 1), x_test[0, :], s=0.2)

    for changepoint in np.nditer(np.argwhere(y_test[0, :])):
        plt.axvline(x=changepoint + 1, color="r", linestyle="--")

    for changepoint in np.nditer(np.argwhere(y_pred[0, :] >= 0.163)):
        plt.axvline(x=changepoint + 1, color="b", linestyle="--")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()
    exit()
