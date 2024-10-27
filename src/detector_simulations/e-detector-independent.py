import matplotlib.pyplot as plt
import numpy as np

alpha = 0.001
num_rounds = 100
timeseries_length = 100
changepoint = 20

changes_found = []
for round in range(num_rounds):
    print(round)

    timeseries = []

    mean_xy = np.zeros(timeseries_length)
    mean_x = np.zeros(timeseries_length)
    mean_y = np.zeros(timeseries_length)
    mean_xy_swap = np.zeros(timeseries_length)

    witness = np.zeros(timeseries_length)
    witness_swap = np.zeros(timeseries_length)
    witness_curr = np.zeros(timeseries_length)
    witness_swap_curr = np.zeros(timeseries_length)

    wealth = np.ones(timeseries_length)
    fractions = np.ones(timeseries_length) * 0.1
    a = np.ones(timeseries_length)

    change_found = False
    for t in range(timeseries_length):
        if t < changepoint:
            timeseries.append((np.random.normal(0, 1), np.random.normal(0, 1) ** 2))
        else:
            x = np.random.normal(0, 1)
            timeseries.append((x, x**2 + np.sin(t)))

        for i in range(t - 2, -1, -2):
            mean_xy[i] = (
                mean_xy[i] * (t - i + 2)
                + timeseries[t - 1][0] * timeseries[t - 1][1]
                + timeseries[t][0] * timeseries[t][1]
            ) / (t - i)

            mean_x[i] = (
                mean_x[i] * (t - i + 2) + timeseries[t - 1][0] + timeseries[t][0]
            ) / (t - i)

            mean_y[i] = (
                mean_y[i] * (t - i + 2) + timeseries[t - 1][1] + timeseries[t][1]
            ) / (t - i)

            mean_xy_swap[i] = (
                mean_xy_swap[i] * (t - i + 2)
                + timeseries[t - 1][0] * timeseries[t][1]
                + timeseries[t][0] * timeseries[t - 1][1]
            ) / (t - i)

            witness[i] = witness_curr[i]
            witness_curr[i] = np.sign(mean_xy[i] - mean_x[i] * mean_y[i])
            witness_swap[i] = witness_swap_curr[i]
            witness_swap_curr[i] = np.sign(mean_xy_swap[i] - mean_x[i] * mean_y[i])

        for i in range(0, t, 2):
            hsic_payoff = (
                witness[i] + witness_curr[i] - witness_swap[i] - witness_swap_curr[i]
            ) / 2

            wealth[i] = wealth[i] * (1 + fractions[i] * hsic_payoff)

            if wealth[i] >= 1 / alpha and not change_found:
                change_found = True
                changes_found.append(t)
                # break

            # z = hsic_payoff / (1 - fractions[i] * hsic_payoff)

            # a[i] += z**2

            # fractions[i] = min(
            #     0.5, max(0, fractions[i] - (2 / (2 - np.log(3)) * z / a[i]))
            # )

        # if change_found:
        #     break


plt.hist(changes_found)

# plt.plot([x[0] for x in timeseries], label="x")
# plt.plot([x[1] for x in timeseries], label="y")
# plt.axvline(x=changes_found[0], color="r", linestyle="--")
plt.show()
