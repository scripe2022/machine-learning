import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
d = np.random.rand(2, 3) < 0.5
print(d)
print(a * d)
