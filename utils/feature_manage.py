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
def extract_labels(paths):
    instance = [int(re.search('/o(.+?)/', path).group(1)) for path in paths]
    session = [int(re.search('s(.+?)/', path).group(1)) for path in paths]
    # 0 indexing
    instance = [i - 1 for i in instance]
    session = [s - 1 for s in session]
    category = [(i - 1) // 5.0 for i in instance]

    instance = np.array(instance)
    session = np.array(session)
    category = np.array(category)

    return (instance, category, session)


@sentinel
def train_extractor(dataset, labels, epochs):
    (instance, category, session) = labels
    num_classes = 50
    BATCH_SIZE = 32
    SPLIT_SIZE = BATCH_SIZE * 2

    # prepare train and validation sets
    train_idxs = [i for i, v in enumerate(session) if v not in [3, 7, 10]]
    validation_idxs = [i for i, v in enumerate(session) if v in [3, 7, 10]]
    train_x = dataset[train_idxs]
    train_y = instance[train_idxs]
    validation_x = dataset[validation_idxs]
    validation_y = instance[validation_idxs]

    # create new model
    SHAPE = (128, 128, 3)
    vgg = tf.keras.applications.VGG16(
        input_shape=SHAPE, include_top=False, weights='imagenet')
    vgg.trainable = False

    extractor = tf.keras.Sequential([
        vgg,
        tf.keras.layers.Conv2D(256, [4, 4])
    ])

    supervised = tf.keras.Sequential([
        extractor,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(num_classes, [1, 1], activation='softmax'),
        tf.keras.layers.Flatten()
    ])

    vgg.summary()
    extractor.summary()
    supervised.summary()

    print('compiling')
    supervised.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    print('done')
    std = train_x.std()

    val_onehot_y = tf.keras.utils.to_categorical(validation_y, num_classes=num_classes)

    train = zip(train_x, train_y)

    splits = math.ceil(len(train_y) / SPLIT_SIZE)
    for split in np.array_split(train, splits):
        (tx, ty) = split
        tx = tx.astype('float')
        tx /= 255.0
        tx /= std

        onehot_y = tf.keras.utils.to_categorical(ty, num_classes=num_classes)
        supervised.fit(
            tx,
            onehot_y,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            shuffle=True,
            validation_data=(validation_x, val_onehot_y))

    return extractor


@sentinel
def extract_features(dataset, extractor):
    print('extracting features')
    extractor.summary()

    features = extractor.predict_on_batch(dataset)
    features = np.squeeze(features)
    print(features.shape())

    return features


@sentinel
def save_dataset(features, labels, path):
    (instance, category, session) = labels
    np.savez(path, x=features, instance=instance, category=category, session=session)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='core50 feature extraction and preprocessing')
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
    parser.add_argument(
        '--epochs',
        default=1,
        help='number of epochs in extractor training')
    parser.add_argument(
        '--save_extractor',
        action='store_true',
        help='save the extractor model')

    args = parser.parse_args()

    paths = load_paths(args.paths)
    dataset = load_core50(args.images, paths)
    labels = extract_labels(paths)

    feature_extractor = train_extractor(dataset, labels, epochs=args.epochs)
    if args.save_extractor:
        feature_extractor.save('extractor_' + date.today() + '.tf')

    features = extract_features(dataset, feature_extractor)
    save_dataset(features, labels, args.out)
