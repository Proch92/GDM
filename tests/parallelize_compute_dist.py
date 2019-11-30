import numpy as np
import time
from multiprocessing import shared_memory


alphas = np.array([0.8, 0.4, 0.2])


def compute_distance(x, input_vector) -> float:
    return np.linalg.norm(np.dot(alphas, (x - input_vector)))


weights = np.random.rand(5000, 3, 512)
input_vector = np.random.rand(3, 512)

t0 = time.time()
a = [compute_distance(w, input_vector) for w in weights]
t1 = time.time()
print(t1 - t0)

t0 = time.time()
b = np.linalg.norm(np.dot(alphas, (weights - input_vector)), axis=1)
t1 = time.time()
print(t1 - t0)

t0 = time.time()
# c = (weights - input_vector)
c = [w - input_vector for w in weights]
c = np.dot(alphas, c)
c = np.linalg.norm(c, axis=1)
t1 = time.time()
print(t1 - t0)

a = np.random.rand()
t0 = time.time()
shm = shared_memory.SharedMemory(create=True, size=weights.nbytes)
t1 = time.time()
print(t1 - t0)
