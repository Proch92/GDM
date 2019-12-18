import pickle
import re
import numpy as np
import random
import os
import imageio
import tensorflow as tf


def batches(iterable, n):
    ln = len(iterable)
    for ndx in range(0, ln, n):
        yield iterable[ndx:min(ndx + n, ln)]


class Core50_Dataset():
    def __init__(self, base_path, paths_file):
        self.base_path = base_path
        self.paths_file = paths_file
        self.load_labels()
        self.split_train_test()

    def __len__(self):
        return self.num_samples

    def load_labels(self):
        pkl_file = open(self.paths_file, 'rb')
        self.paths = pickle.load(pkl_file)
        self.paths = self.paths[::4]
        self.num_samples = len(self.paths)

        instance = [int(re.search('/o(.+?)/', path).group(1)) for path in self.paths]
        session = [int(re.search('s(.+?)/', path).group(1)) for path in self.paths]

        instance = [i - 1 for i in instance]
        session = [s - 1 for s in session]
        category = [(i - 1) // 5.0 for i in instance]

        self.instance = np.array(instance)
        self.session = np.array(session)
        self.category = np.array(category)

    def load(self, paths):
        dataset = np.ndarray((len(paths), 128, 128, 3), np.uint8)
        for i in range(len(paths)):
            dataset[i] = imageio.imread(os.path.join(self.base_path, paths[i]))
        return dataset

    def shuffle(self):
        z = list(zip(self.paths, self.instance, self.session, self.category))
        random.shuffle(z)
        self.paths, self.instance, self.session, self.category = zip(*z)

    def split_train_test(self):
        train_idx = [idx for idx in range(self.num_samples) if self.session[idx] not in [3, 7, 10]]
        test_idx = [idx for idx in range(self.num_samples) if self.session[idx] in [3, 7, 10]]

        self.train_paths = np.take(self.paths, train_idx, axis=0)
        self.train_y = np.take(self.instance, train_idx, axis=0)
        self.test_paths = np.take(self.paths, test_idx, axis=0)
        self.test_y = np.take(self.instance, test_idx, axis=0)

        self.train_len = len(self.train_paths)
        self.test_len = len(self.test_paths)

    def x_gen(self, batch_size):
        for batch in batches(self.paths, batch_size):
            x = self.load(batch)
            x = x.astype('float32')
            x /= 255.0
            yield x

    def train_gen_forever(self, batch_size):
        while True:
            for val in self.train_gen(batch_size):
                yield val

    def train_gen(self, batch_size, shuffle=True):
        dataset = list(zip(self.train_paths, self.train_y))
        batch_list = list(batches(dataset, batch_size))
        if shuffle:
            random.shuffle(batch_list)
        for batch in batch_list:
            paths, y = zip(*batch)
            x = self.load(paths)
            x = x.astype('float32')
            x /= 255.0
            y = tf.keras.utils.to_categorical(y, num_classes=50)
            yield (x, y)

    def test(self, sample=1):
        sample_population = int(len(self.test_paths) * sample)
        sampled = random.sample(list(range(len(self.test_paths))), sample_population)

        paths = self.test_paths[sampled]
        y = self.test_y[sampled]

        x = self.load(paths)
        x = x.astype('float32')
        x /= 255.0
        y = tf.keras.utils.to_categorical(y, num_classes=50)
        return (x, y)

