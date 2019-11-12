from collections import deque
from multiprocessing import Process, Pipe
import publish
from functools import partial
import time

import matplotlib.pyplot as plt
plt.ion()

processes = []
pipes = {}


def plotter(pipe, max_len, refresh_rate):
    buffer = deque([0.0] * max_len)

    xlim = (0, max_len)
    ylim = (0, 100)

    fig = plt.figure()
    plt.axes(xlim=xlim, ylim=ylim)
    line, = plt.plot(buffer)

    while(True):
        val = pipe.recv()
        while pipe.poll():
            val = pipe.recv()

        if val < ylim[0]:
            ylim = (val, ylim[1])
            plt.ylim(ylim)
        if val > ylim[1]:
            ylim = (ylim[0], val)
            plt.ylim(ylim)

        buffer.popleft()
        buffer.append(val)
        line.set_ydata(buffer)
        fig.canvas.draw()
        time.sleep(refresh_rate)


def pipe_send(val, topic):
    pipes[topic].send(val)


def plot(topic, max_len=1000, refresh_rate=1):
    child_conn, parent_conn = Pipe(duplex=False)
    p = Process(target=plotter, args=(child_conn, max_len, refresh_rate))
    p.daemon = True
    p.start()
    processes.append(p)
    pipes[topic] = parent_conn

    publish.subscribe(partial(pipe_send, topic=topic), topic)
