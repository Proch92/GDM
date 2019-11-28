import publish
import pickle
import time


class Profiler(object):
    def __init__(self):
        self.buffers = {}
        publish.subscribe_to_all(self.on_receive)

    def on_receive(self, topic, val):
        if topic not in self.buffers.keys():
            self.buffers[topic] = []
        self.buffers[topic].append(val)

    def save_all(self, name):
        with open('profiling/' + name + '.pkl', 'wb') as f:
            pickle.dump(self.buffers, f)


def timeit(foo):
    def timed(*args, **kw):
        ts = time.time()
        result = foo(*args, **kw)
        te = time.time()

        print('%r %f sec' % (foo.__name__, te - ts))
        return result
    return timed
