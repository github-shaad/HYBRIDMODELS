import numpy as np


arr = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
print(np.vstack((arr.reshape(-1, 1), arr2.reshape(-1, 1))))