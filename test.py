import numpy as np

m = np.matrix([[1, 2], [3, 4]])

a1 = np.array(m)
a2 = np.asanyarray(m)

print(type(a1))  # <class 'numpy.ndarray'>
print(type(a2))  # <class 'numpy.matrix'>
print(a1.shape)
