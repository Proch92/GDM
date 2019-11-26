import publish
from functools import partial
import pickle
import time


class Profiler(object):
    def __init__(self, topics):
        self.topics = topics
        self.buffers = {}

        for topic in self.topics:
            self.buffers[topic] = []
            publish.subscribe(partial(self.on_receive, topic=topic), topic)

    def on_receive(self, val, topic):
        self.buffers[topic].append(val)

    def save_all(self):
        for topic in self.topics:
            with open('profiling/topics/' + topic + '.pkl', 'wb') as f:
                pickle.dump(self.buffers[topic], f)


def timeit(foo):
    def timed(*args, **kw):
        ts = time.time()
        result = foo(*args, **kw)
        te = time.time()

        print('%r %f sec' % (foo.__name__, te - ts))
        return result
    return timed
