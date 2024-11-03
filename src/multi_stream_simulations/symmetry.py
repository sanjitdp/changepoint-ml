import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

alpha = 0.01
num_streams = 50
num_rounds = 100
timeseries_length = 1000
changepoint = 200

changes_found = []

for _ in tqdm(range(num_rounds)):
    x = np.random.normal(size=[num_streams, timeseries_length])
    x[changepoint:, :] = np.exp(x[changepoint:, :]) - np.exp(-0.5)
    e_processes = np.zeros((num_streams, timeseries_length))
    wealth = np.zeros(num_streams)

    change_found = False
    for t in range(timeseries_length):
        for k in range(num_streams):
            curr_obs = x[k, t]
            e_processes[k, t] = 1
            e_processes[k, :] += np.sign(curr_obs) * (e_processes[k, :] > 0)

            wealth[k] = np.max(e_processes[k, :])

        if change_found:
            break

plt.hist(changes_found, bins=100)

plt.show()
