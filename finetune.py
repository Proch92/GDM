import tensorflow as tf
import numpy as np
import pandas as pd
import math


BATCH_SIZE = 32
SPLIT_SIZE = BATCH_SIZE * 100
EPOCHS = 20
num_classes = 50

# load and preprocess images
with np.load('core50/core50_imgs_5fps.npz') as core50_5fps:
    imgs = core50_5fps['x']
    instance = core50_5fps['instance']
    session = core50_5fps['session']

ds = pd.DataFrame({
    'x': range(len(imgs)),
    'instance': instance,
    'session': session
})

train = ds[ds['session'].isin([1, 2, 4, 5, 6, 8, 9, 11])]

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

supervised.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# training
for epoch in range(EPOCHS):
    print('--------------------- epoch: {}/{} ----------------'.format(epoch, EPOCHS))

    # shuffle data
    train = train.sample(frac=1)

    for split in np.array_split(train, math.ceil(len(train) / SPLIT_SIZE)):
        split_indexes = split['x'].values
        batch_x = imgs[split_indexes]
        batch_x = batch_x / 255.0

        batch_y = split['instance'].values
        batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=num_classes)

        supervised.fit(batch_x, batch_y, batch_size=BATCH_SIZE, epochs=1)

extractor.save('extractor3linear.tf')
