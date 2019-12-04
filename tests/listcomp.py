import numpy as np
import timeit


fix = np.array([0.8, 0.4, 0.2])


def compute(x, b):
    return np.linalg.norm(np.dot(fix, (x - b)))


mat = np.random.rand(2500, 3, 512)
b = np.random.rand(3, 512)


def work():
    np.array([compute(w, b) for w in mat])


t = timeit.timeit(work, number=100)
print(t / 100)
