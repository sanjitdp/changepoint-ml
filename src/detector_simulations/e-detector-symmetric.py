import matplotlib.pyplot as plt
import numpy as np

alpha = 0.01
num_rounds = 1000
timeseries_length = 1000
changepoint = 200

changes_found = []
for round in range(num_rounds):
    print(round)
    e_processes = np.zeros(timeseries_length)

    change_found = False
    for t in range(timeseries_length):
        curr_obs = (
            np.random.normal(0, 1)
            if t < changepoint
            else np.exp(np.random.normal(0, 1)) - np.exp(-0.5)
        )

        e_processes[t] = 1
        e_processes += np.sign(curr_obs) * (e_processes > 0)

        if np.any(e_processes >= 1 / alpha) and not change_found:
            change_found = True
            changes_found.append(t)

        if change_found:
            break

plt.hist(changes_found, bins=100)

# plt.plot(timeseries)
# plt.axvline(x=change_found, color="r", linestyle="--")
plt.show()
