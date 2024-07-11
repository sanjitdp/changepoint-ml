import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import optim
from models.feedforward_earthmover_network import FeedforwardEarthmoverNetwork
from time import time

TIMESERIES_LENGTH = 1024
TRAIN_EXAMPLES = 8192
TEST_EXAMPLES = 2048
CHANGEPOINT_PROB = 0.01
POSSIBLE_MEANS = list(range(-3, 4))
GAUSSIAN_VARIANCE = 1
NUM_EPOCHS = 1000
MODEL = FeedforwardEarthmoverNetwork
LOAD_STATE = "src/trained_models/feedforward_earthmover_network_weights_1024.pth"  # set to None to train from scratch
SAVE_STATE = "src/trained_models/feedforward_earthmover_network_weights_1024.pth"  # set to None to not save weights
PLOT = False

if PLOT:
    TRAIN_EXAMPLES = 1

very_start_time = time()

print("Generating training data...")

start_time = time()

x_train = np.zeros((TRAIN_EXAMPLES, TIMESERIES_LENGTH))
y_train = np.zeros((TRAIN_EXAMPLES, TIMESERIES_LENGTH))

curr_mean = 0
for i in range(TRAIN_EXAMPLES):
    for j in range(TIMESERIES_LENGTH):
        coin = np.random.choice([0, 1], p=[1 - CHANGEPOINT_PROB, CHANGEPOINT_PROB])
        if coin:
            new_mean = np.random.choice(POSSIBLE_MEANS)
            if new_mean != curr_mean:
                y_train[i][j] = 1
                curr_mean = new_mean

        x_train[i][j] = np.random.normal(curr_mean, GAUSSIAN_VARIANCE)

end_time = time()

print("Data generated!")
print(f"Time taken: {(end_time - start_time):.2f} seconds")

if PLOT:
    plt.title("Gaussian time series with mean change")
    plt.scatter(range(1, TIMESERIES_LENGTH + 1), x_train[0, :], s=0.2)

    for changepoint in np.nditer(np.argwhere(y_train[0, :])):
        plt.axvline(x=changepoint + 1, color="r", linestyle="--")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()
    exit()

print()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = MODEL(TIMESERIES_LENGTH).to(device)
if LOAD_STATE:
    model.load_state_dict(torch.load(LOAD_STATE))
    print("Model weights loaded!")

model.train()

criterion = model.earthmover_loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

x_tensor = torch.from_numpy(x_train).float().to(device)
y_tensor = torch.from_numpy(y_train).float().to(device)

print()

print("Training model...")
start_time = time()
for epoch in range(NUM_EPOCHS):
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)

    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        end_time = time()
        print(f"------------------\nEpoch {epoch+1}/{NUM_EPOCHS}\n------------------")
        print(f"Mean training loss: {loss.sum().item()}")
        print(f"Time taken: {(end_time - start_time):.2f} seconds")
        print()
        start_time = end_time

print("Generating testing data...")

start_time = time()
x_test = np.zeros((TEST_EXAMPLES, TIMESERIES_LENGTH))
y_test = np.zeros((TEST_EXAMPLES, TIMESERIES_LENGTH))

curr_mean = 0
for i in range(TEST_EXAMPLES):
    for j in range(TIMESERIES_LENGTH):
        coin = np.random.choice([0, 1], p=[1 - CHANGEPOINT_PROB, CHANGEPOINT_PROB])
        if coin:
            new_mean = np.random.choice(POSSIBLE_MEANS)
            if new_mean != curr_mean:
                y_test[i][j] = 1
                curr_mean = new_mean

        x_test[i][j] = np.random.normal(curr_mean, GAUSSIAN_VARIANCE)

x_test_tensor = torch.from_numpy(x_test).float().to(device)
y_test_tensor = torch.from_numpy(y_test).float().to(device)

print("Testing data generated!")
print(f"Time taken: {(time() - start_time):.2f} seconds")

print()

model.eval()
with torch.no_grad():
    test_loss = criterion(model(x_test_tensor), y_test_tensor).mean().item()
    print(f"Mean testing loss: {test_loss}")
    print()

if SAVE_STATE:
    torch.save(model.state_dict(), SAVE_STATE)
    print("Model weights saved!")

print()

print(f"Total time taken: {(time() - very_start_time):.2f} seconds")

print()
