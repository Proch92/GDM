callbacks = {}
callbacks_all = []


def subscribe(callback, topic):
    if topic not in callbacks.keys():
        callbacks[topic] = []
    callbacks[topic].append(callback)


def subscribe_to_all(callback):
    callbacks_all.append(callback)


def send(topic, message):
    [c(topic, message) for c in callbacks_all]
    if topic in callbacks.keys():
        [c(message) for c in callbacks[topic]]
