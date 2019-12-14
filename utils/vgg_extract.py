import argparse
import tensorflow as tf
from datetime import date
import pickle
import numpy as np
import re
import os
from tqdm import tqdm
import imageio
import random
import math


def sentinel(foo):
    def wrapper(*args, **kwargs):
        print(foo.__name__)
        return foo(*args, **kwargs)
    return wrapper


@sentinel
def load_paths(path, shuffle=True):
    pkl_file = open(path, 'rb')
    paths = pickle.load(pkl_file)
    paths = paths[::4]
    random.shuffle(paths)
    return paths


@sentinel
def load_core50(core50_path, paths):
    dataset = np.ndarray((len(paths), 128, 128, 3), np.uint8)

    for i in tqdm(range(len(paths))):
        dataset[i] = imageio.imread(os.path.join(core50_path, paths[i]))

    print(f'loaded core50. dataset shape {dataset.shape} and dtype {dataset.dtype}')
    return dataset


@sentinel
def get_extractor():
    # create new model
    SHAPE = (128, 128, 3)
    vgg = tf.keras.applications.VGG16(
        input_shape=SHAPE, include_top=False, weights='imagenet')

    return vgg


@sentinel
def extract_features(dataset, extractor):
    extractor.summary()

    features = []
    indexes = list(range(len(dataset)))
    for split in np.array_split(indexes, 1000):
        dsplit = dataset[split]
        dsplit = dsplit.astype('float')
        dsplit /= 255.0
        preds = extractor.predict_on_batch(dsplit)
        for pred in preds:
            pred = np.squeeze(pred)
            features.append(pred)

    features = np.array(features)
    print(features.shape)

    return features


@sentinel
def save_dataset(features, path):
    np.savez(path, x=features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vgg16 feature extraction')
    parser.add_argument(
        '--images',
        required=True,
        help='specify core50 images path')
    parser.add_argument(
        '--out',
        required=True,
        help='output dataset file path')
    parser.add_argument(
        '--paths',
        required=True,
        help='paths.pkl file')

    args = parser.parse_args()

    paths = load_paths(args.paths)
    dataset = load_core50(args.images, paths)

    feature_extractor = get_extractor()

    features = extract_features(dataset, feature_extractor)
    save_dataset(features, args.out)
