import numpy as np
import time
from multiprocessing import shared_memory
from multiprocessing import Process
from multiprocessing import Pool
from functools import partial


alphas = np.array([0.8, 0.4, 0.2])


def compute_distance(x, input_vector):
    return np.linalg.norm(np.dot(alphas, (x - input_vector)))


weights = np.random.rand(2500, 3, 512)
input_vector = np.random.rand(3, 512)

# list comp w/ function call
t0 = time.time()
a = [compute_distance(w, input_vector) for w in weights]
t1 = time.time()
print(t1 - t0)

# broadcast
t0 = time.time()
b = np.linalg.norm(np.dot(alphas, (weights - input_vector)), axis=1)
t1 = time.time()
print(t1 - t0)

# multiprocessing w/ pools
pool = Pool(8)
t0 = time.time()
comp = partial(compute_distance, input_vector=input_vector)
pool.map(comp, weights)
t1 = time.time()
print(t1 - t0)
pool.close()


# multiprocessing w/ shared memory
def compute_distance_shm(idx, input_vector, shm, shape, dtype, chunk_size):
    t0 = time.time()
    ws = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    for i in range(idx * chunk_size, (idx + 1) * chunk_size):
        np.linalg.norm(np.dot(alphas, (ws[i] - input_vector)))
    t1 = time.time()
    print('process_', idx, ' ', t1 - t0)


n_workers = 8
t0 = time.time()
shm = shared_memory.SharedMemory(create=True, size=weights.nbytes)
t1 = time.time()
print('shm creation ', t1 - t0)

t0 = time.time()
chunk_size = weights.shape[0] // n_workers
comp = partial(compute_distance_shm, input_vector=input_vector, shm=shm, shape=weights.shape, dtype=weights.dtype, chunk_size=chunk_size)

procs = []
for i in range(n_workers):
    procs.append(Process(target=comp, args=(i,)))
    procs[i].start()

for p in procs:
    p.join()

t1 = time.time()
print(t1 - t0)
shm.close()
shm.unlink()
