import numpy as np
import matplotlib.pyplot as plt

# Random definition of used parameters:
lam = 1.2  # Spreading rate
sigma = 0.2  # Incubation period
mi = 0.15  # Recovery rate
recovery_rate = lam / mi

change_point_1 = 30
change_point_2 = 75 # TODO change_points
report_delay = 8

N = 1000  # Population size
T = 150 # No. of days (timepoints)

# No. of people at t = 0 (at the outbreak of pandemic)
susceptible_array = np.zeros(T + report_delay)
exposed_array = np.zeros(T + report_delay)
infected_array = np.zeros(T + report_delay)
recovered_array = np.zeros(T + report_delay)

susceptible_array[0] = N - 40
exposed_array[0] = 20  # Paper
infected_array[0] = 20  # Paper
recovered_array[0] = 0

for t in range(1, T):
    susceptible_array[t] = susceptible_array[t-1] - (lam * susceptible_array[t-1] * infected_array[t-1]) / N
    exposed_array[t] = exposed_array[t-1] + (lam * susceptible_array[t-1] * infected_array[t-1]) / N - sigma * exposed_array[t-1]
    infected_array[t] = infected_array[t-1] + sigma * exposed_array[t - 1] - mi * infected_array[t-1]
    recovered_array[t] = recovered_array[t-1] + mi * infected_array[t - 1]
    print(susceptible_array[t] + exposed_array[t] + infected_array[t] + recovered_array[t])

days = range(T)
plt.plot(days, susceptible_array, label='susceptible')
plt.plot(days, exposed_array, label='exposed')
plt.plot(days, infected_array, label='infected')
plt.plot(days, recovered_array, label='recovered')
leg = plt.legend(loc='upper right')
plt.show()


