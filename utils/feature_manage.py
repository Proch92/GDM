import argparse
import tensorflow as tf
from datetime import date
import numpy as np
import math
from core50_loader import Core50_Dataset


gpus = tf.config.experimental.list_logical_devices('GPU')
use_specific_gpu = 1


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
        tf.keras.layers.Conv2D(256, [1, 1], activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(256, [4, 4])
    ])

    supervised = tf.keras.Sequential([
        extractor,
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(num_classes, [1, 1], activation='softmax'),
        tf.keras.layers.Flatten()
    ])

    supervised.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    supervised.summary()

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
    batch_size = 32
    features = None
    for batch in dataset.x_gen(batch_size):
        preds = extractor.predict_on_batch(batch)
        preds = np.squeeze(preds)
        if features is None:
            features = preds
        else:
            features = np.vstack((features, preds))

    print(features.shape)
    std = features.std()
    features /= std

    return features


@sentinel
def save_dataset(features, dataset, path):
    instance = dataset.instance
    category = dataset.category
    session = dataset.session
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

    def train_and_extract():
        feature_extractor = train_extractor(dataset, epochs=args.epochs)
        if args.save_extractor:
            feature_extractor.save('extractor_' + str(date.today()) + '.tf')
        return extract_features(dataset, feature_extractor)

    if len(gpus) > 0 and use_specific_gpu >= 0:
        with tf.device(gpus[use_specific_gpu].name):
            features = train_and_extract()
    else:
        features = train_and_extract()

    save_dataset(features, dataset, args.out)

