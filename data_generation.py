import numpy as np
import matplotlib.pyplot as plt

# Random definition of used parameters:
lam = np.array([1.1, 0.2, 0.3])  # Spreading rate
sigma = 0.2  # Incubation period
mi = 0.15  # Recovery rate
# recov = 1/100
# recovery_rate = lam / mi

change_point_1 = 19
change_point_2 = 30
report_delay = 8

N = 1000  # Population size
T = 65 # No. of days (timepoints)

# No. of people at t = 0 (at the outbreak of pandemic)
susceptible_array = np.zeros(T)
exposed_array = np.zeros(T)
infected_array = np.zeros(T)
recovered_array = np.zeros(T)

susceptible_array[0] = N - 40
exposed_array[0] = 10  # Paper
infected_array[0] = 2  # Paper
recovered_array[0] = 0

for t in range(1, T):
    if t < change_point_1:
        change = 0
    elif t < change_point_2:
        change = 1
    else:
        change = 2
    susceptible_array[t] = susceptible_array[t-1] - (lam[change] * susceptible_array[t-1] * infected_array[t-1]) / N # + recovered_array[t-1] * recov
    exposed_array[t] = exposed_array[t-1] + (lam[change] * susceptible_array[t-1] * infected_array[t-1]) / N - sigma * exposed_array[t-1]
    infected_array[t] = infected_array[t-1] + sigma * exposed_array[t - 1] - mi * infected_array[t-1]
    recovered_array[t] = recovered_array[t-1] + mi * infected_array[t - 1] # - recovered_array[t-1] * recov
    # print(susceptible_array[t] + exposed_array[t] + infected_array[t] + recovered_array[t])

case_statistics_array = np.zeros(T)
for i in range(report_delay, T):
    case_statistics_array[i] = infected_array[i - report_delay]

days = range(T)
plt.plot(days, susceptible_array, label='susceptible')
plt.plot(days, exposed_array, label='exposed')
plt.plot(days, infected_array, label='infected')
plt.plot(days, recovered_array, label='recovered')

plt.plot(days, case_statistics_array, label='reported cases')
leg = plt.legend(loc='upper right')
plt.show()


