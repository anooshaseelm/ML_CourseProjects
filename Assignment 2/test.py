import numpy as np

x = np.zeros((2,4))
y = np.ones((2,4))
all_data = np.concatenate((x, y), 1)

print(all_data)