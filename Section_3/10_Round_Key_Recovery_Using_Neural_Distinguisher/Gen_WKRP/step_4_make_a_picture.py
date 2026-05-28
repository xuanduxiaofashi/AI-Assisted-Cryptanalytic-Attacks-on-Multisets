import numpy as np
import matplotlib.pyplot as plt
import re


N = 16
mean_matrix = np.zeros((N, N))
std_matrix = np.zeros((N, N))

with open('output.txt', 'r') as f:
    text = f.read()

pattern = r"(\d+)_(\d+), mean: ([\d.]+), std\s*:\s*([\d.]+)"

for match in re.finditer(pattern, text):
    row, col, mean, std = match.groups()
    row = int(row)
    col = int(col)
    mean_matrix[row][col] = float(mean)
    std_matrix[row][col] = float(std)


print("Mean matrix:\n", mean_matrix)
print("Std matrix:\n", std_matrix)


mean_matrix_reshape = np.zeros((N, N))
std_matrix_reshape = np.zeros((N, N))

for i in range(16):
    for j in range(16):
        mean_matrix_reshape[i ^ 11][j ^ 12] = mean_matrix[i][j]
        std_matrix_reshape[i ^ 11][j ^ 12] = std_matrix[i][j]


flattened_mean = mean_matrix_reshape.flatten()
x = np.arange(len(flattened_mean))
plt.figure(figsize=(10, 4))
plt.plot(x, flattened_mean)
plt.xlabel('Difference to real key')
plt.ylabel('Mean response')

plt.savefig('mean_matrix_plot.png', dpi=300, bbox_inches='tight')


flattened_std = std_matrix_reshape.flatten()
x = np.arange(len(flattened_std))
plt.figure(figsize=(10, 4))
plt.plot(x, flattened_std)
plt.xlabel('Difference to real key')
plt.ylabel('Std response')
plt.savefig('std_matrix_plot.png', dpi=300, bbox_inches='tight')


np.save('average_list_diff.npy', mean_matrix_reshape)
np.save('std_list_diff.npy', std_matrix_reshape)