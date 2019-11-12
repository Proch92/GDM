callbacks = {}


def subscribe(callback, topic):
    if topic not in callbacks.keys():
        callbacks[topic] = []
    callbacks[topic].append(callback)


def send(topic, message):
    if topic in callbacks.keys():
        [c(message) for c in callbacks[topic]]
