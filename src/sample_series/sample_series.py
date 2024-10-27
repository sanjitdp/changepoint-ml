import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import optim
from models.feedforward_earthmover_network import FeedforwardEarthmoverNetwork
from models.feedforward_cusum_network import FeedforwardCUSUMNetwork
from time import time

TIMESERIES_LENGTH = 8192
TRAIN_EXAMPLES = 1
TEST_EXAMPLES = 64
CHANGEPOINT_PROB = 0.0005
POSSIBLE_MEANS = list(range(-3, 4))
GAUSSIAN_VARIANCE = 1
NUM_EPOCHS = 1000
MODEL = FeedforwardCUSUMNetwork
LOAD_STATE = None  # set to None to train from scratch
SAVE_STATE = None  # set to None to not save weights
PLOT = True

if PLOT:
    TRAIN_EXAMPLES = 1

very_start_time = time()

print("Generating training data...")

start_time = time()

x_train = np.zeros((TRAIN_EXAMPLES, TIMESERIES_LENGTH))
y_train = np.zeros((TRAIN_EXAMPLES, TIMESERIES_LENGTH))

curr_mean = 0
log_concave = True
for i in range(TRAIN_EXAMPLES):
    for j in range(TIMESERIES_LENGTH):
        coin = np.random.choice([0, 1], p=[1 - CHANGEPOINT_PROB, CHANGEPOINT_PROB])
        if coin:
            log_concave = not log_concave
            y_train[i][j] = 1

        x_train[i][j] = np.random.normal(curr_mean, GAUSSIAN_VARIANCE)
        if not log_concave:
            x_train[i][j] = np.exp(x_train[i][j]) - np.exp(-0.5)

end_time = time()

print("Data generated!")
print(f"Time taken: {(end_time - start_time):.2f} seconds")

plt.title("Example time series with symmetricity change")
plt.scatter(range(1, TIMESERIES_LENGTH + 1), x_train[0, :], s=0.2)

for changepoint in np.nditer(np.argwhere(y_train[0, :])):
    plt.axvline(x=changepoint + 1, color="r", linestyle="--")

plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
