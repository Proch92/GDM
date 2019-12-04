import numpy as np
import timeit


fix = np.array([0.8, 0.4, 0.2])
mat = np.random.rand(2500, 3, 512)
b = np.random.rand(3, 512)


def work():
    np.linalg.norm(np.dot(fix, (mat - b)), axis=1)


t = timeit.timeit(work, number=100)
print(t / 100)
