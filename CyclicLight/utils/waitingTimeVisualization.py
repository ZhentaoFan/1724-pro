import matplotlib.pyplot as plt
import re

# Extracting the data from the file content
data = re.findall(r'round (\d+): (\d+\.?\d*)', file_content)

# Parsing the extracted data
rounds, differences = zip(*[(int(r), float(d)) for r, d in data])

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(rounds, differences, marker='o', color='b')
plt.title('Total Waiting Time per Round')
plt.xlabel('Round')
plt.ylabel('Waiting Time')
plt.grid(True)
plt.show()
