import pickle
import re
import numpy as np
import random
import math
import os
import imageio
import tensorflow as tf


class Core50_Dataset():
    def __init__(self, base_path, paths_file):
        self.base_path = base_path
        self.paths_file = paths_file
        self.load_labels()
        self.split_train_test()

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
        z = zip(self.paths, self.instance, self.session, self.category)
        random.shuffle(z)
        self.paths, self.instance, self.session, self.category = zip(*z)

    def split_train_test(self):
        train_idx = [idx for idx in range(self.num_samples) if self.session[idx] not in [3, 7, 10]]
        test_idx = [idx for idx in range(self.num_samples) if self.session[idx] in [3, 7, 10]]

        self.train_paths = self.paths[train_idx]
        self.train_y = self.instance[train_idx]
        self.test_paths = self.paths[test_idx]
        self.test_y = self.instance[test_idx]

    def train(self, batch_size):
        print(f'expected batch size: {128 * 128 * 3 * 32 * batch_size} bytes')
        num_batches = math.ceil(self.num_samples / batch_size)
        for batch in np.array_split(zip(self.train_paths, self.train_y), num_batches):
            paths, y = batch
            x = self.load(paths)
            x = x.astype('float32')
            x /= 255.0
            y = tf.keras.utils.to_categorical(y, num_classes=50)
            yield (x, y)
