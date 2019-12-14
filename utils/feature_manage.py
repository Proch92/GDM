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
    category = [i // 5.0 for i in instance]

    instance = np.array(instance)
    session = np.array(session)
    category = np.array(category)

    return (instance, category, session)


@sentinel
def train_extractor(dataset, labels, epochs):
    (instance, category, session) = labels
    num_classes = 50
    BATCH_SIZE = 32
    SPLIT_SIZE = BATCH_SIZE * 500

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
        tf.keras.layers.Conv2D(256, [4, 4], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(256, [1, 1], kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ])

    supervised = tf.keras.Sequential([
        extractor,
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.5),
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

    val_onehot_y = tf.keras.utils.to_categorical(validation_y, num_classes=num_classes)

    indexes = list(range(len(train_x)))

    splits = math.ceil(len(indexes) / SPLIT_SIZE)
    for split in np.array_split(indexes, splits):
        tx = train_x[split]
        ty = train_y[split]
        tx = tx.astype('float')
        tx /= 255.0

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
        type=int,
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
        feature_extractor.save('extractor_' + str(date.today()) + '.tf')

    features = extract_features(dataset, feature_extractor)
    save_dataset(features, labels, args.out)
