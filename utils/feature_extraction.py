import argparse
import tensorflow as tf
from datetime import date
import pickle
import numpy as np
import re
import os
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
def load_core50(core50_path, labels):
    (instance, category, session) = labels
    train_idxs = [idx for idx, sess in enumerate(session) if sess not in [3, 7, 10]]
    test_idxs = [idx for idx, sess in enumerate(session) if sess in [3, 7, 10]]
    with np.load(core50_path) as feats:
        train = feats['x'][train_idxs]
        test = feats['x'][test_idxs]

    def preprocess(x, y):
        x = x / 255.0
        return (x, y)

    train_labels = np.array(instance[train_idxs])
    test_labels = np.array(instance[test_idxs])
    # train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=50)
    # test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=50)
    train = tf.data.Dataset.from_tensor_slices((train, train_labels))
    test = tf.data.Dataset.from_tensor_slices((test,  test_labels))
    train = train.map(preprocess)
    test = test.map(preprocess)

    return (train, test)


@sentinel
def extract_labels(paths):
    instance = [int(re.search('/o(.+?)/', path).group(1)) for path in paths]
    session = [int(re.search('s(.+?)/', path).group(1)) for path in paths]
    # 0 indexing
    instance = [i - 1 for i in instance]
    session = [s - 1 for s in session]
    category = [i // 5.0 for i in instance]

    instance = np.array(instance)
    session = np.array(session)
    category = np.array(category)

    return (instance, category, session)


@sentinel
def train_extractor(dataset, epochs):
    (train, test) = dataset
    num_classes = 50
    BATCH_SIZE = 64

    # create new model
    extractor = tf.keras.Sequential([
        tf.keras.layers.Conv2D(256, [4, 4], input_shape=(4, 4, 512), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, [1, 1], kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])

    supervised = tf.keras.Sequential([
        extractor,
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(num_classes, [1, 1], activation='softmax'),
        tf.keras.layers.Flatten()
    ])

    extractor.summary()
    supervised.summary()

    supervised.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    train_batched = train.batch(BATCH_SIZE)
    test_batched = test.batch(BATCH_SIZE)

    supervised.fit(
        train_batched,
        epochs=epochs,
        shuffle=True,
        # validation_data=test_batched
    )

    return extractor


@sentinel
def extract_features(dataset, extractor):
    (train, test) = dataset
    dataset = train.concatenate(test)
    preds = extractor.predict_on_batch(dataset)
    print(features.shape)
    std = features.std()
    features /= std

    return features


@sentinel
def save_dataset(features, labels, path):
    (instance, category, session) = labels
    np.savez(path, x=features, instance=instance, category=category, session=session)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='core50 feature extraction and preprocessing')
    parser.add_argument(
        '--features',
        required=True,
        help='specify core50 VGG extracted features path')
    parser.add_argument(
        '--out',
        required=True,
        help='output dataset file path')
    parser.add_argument(
        '--paths',
        required=True,
        help='paths.pkl file')
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='number of epochs in extractor training')
    parser.add_argument(
        '--save_extractor',
        action='store_true',
        help='save the extractor model')

    args = parser.parse_args()

    paths = load_paths(args.paths)
    labels = extract_labels(paths)
    dataset = load_core50(args.features, labels)

    feature_extractor = train_extractor(dataset, epochs=args.epochs)
    if args.save_extractor:
        feature_extractor.save('extractor_' + str(date.today()) + '.tf')

    features = extract_features(dataset, feature_extractor)
    save_dataset(features, labels, args.out)
