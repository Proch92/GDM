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
from core50_loader import Core50_Dataset


gpus = tf.config.experimental.list_logical_devices('GPU')


def sentinel(foo):
    def wrapper(*args, **kwargs):
        print(foo.__name__)
        return foo(*args, **kwargs)
    return wrapper


@sentinel
def train_extractor(dataset, epochs):
    num_classes = 50
    BATCH_SIZE = 32

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

    supervised.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    dataset.shuffle()
    supervised.fit_generator(
        dataset.train_gen_forever(BATCH_SIZE),
        steps_per_epoch=math.ceil(dataset.train_len / BATCH_SIZE),
        epochs=epochs,
        validation_data=dataset.test(sample=0.3)
    )

    return extractor


@sentinel
def extract_features(dataset, extractor):
    batch_size = 512
    preds = extractor.predict_generator(dataset.x_gen(batch_size), steps=math.ceil(len(dataset) / batch_size))
    preds = np.squeeze(preds)

    print(preds.shape)
    std = preds.std()
    preds /= std

    return preds


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

    dataset = Core50_Dataset(args.images, args.paths)

    with tf.device(gpus[1].name):
        feature_extractor = train_extractor(dataset, epochs=args.epochs)
        if args.save_extractor:
            feature_extractor.save('extractor_' + str(date.today()) + '.tf')

        features = extract_features(dataset, feature_extractor)
    save_dataset(features, labels, args.out)
