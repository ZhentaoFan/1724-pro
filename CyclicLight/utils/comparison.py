import matplotlib.pyplot as plt

# Data for plotting
rounds = [40, 41, 42, 43, 44]

# First set of data
# sum_differences_1 = [889671.0, 896977.0, 893606.0, 894207.0, 891386.0]

# Second set of data
sum_differences_2 = [794509.0, 794077.0, 794878.0, 794637.0, 793472.0]

# Third set of data
sum_differences_3 = [894884.0, 908755.0, 893887.0, 896889.0, 896922.0]

# Plotting
plt.figure(figsize=(10, 6))
# plt.plot(rounds, sum_differences_1, label='CyclicLight')

plt.plot(rounds, sum_differences_3, label='DataLight',color='orange')
plt.plot(rounds, sum_differences_2, label='Non-Cylic Baseline',color='g')
plt.fill_between(rounds, sum_differences_2, sum_differences_3, color='gray', alpha=0.5)

# Adding titles and labels
plt.title('Total Waiting Time for Rounds 40-44')
plt.xlabel('Round Number')
plt.ylabel('Total Waiting Time')
plt.xticks(rounds)
plt.legend()

# Show plot
plt.grid(True)
plt.show()
