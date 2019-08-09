import tensorflow as tf
import numpy as np


BATCH_SIZE = 16
SPLIT_SIZE = BATCH_SIZE * 10
EPOCHS = 10

# load and preprocess images
core50_5fps = np.load('5fps.npz')
imgs = core50_5fps['x']
labels = core50_5fps['y']
del core50_5fps
num_classes = 50

# create new model
SHAPE = (128, 128, 3)
vgg = tf.keras.applications.VGG16(
    input_shape=SHAPE, include_top=False, weights='imagenet')
vgg.trainable = False

extractor = tf.keras.Sequential([
    vgg,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256)
])

supervised = tf.keras.Sequential([
    extractor,
    tf.keras.layers.Dense(num_classes)
])

supervised.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# shuffle
indexes = np.arange(len(labels))
np.random.shuffle(indexes)

# training
for epoch in range(EPOCHS):
    print('--------------------- epoch: {}/{} ----------------'.format(epoch, EPOCHS))
    for batch_idx in [indexes[x: x + SPLIT_SIZE] for x in range(0, len(indexes), SPLIT_SIZE)]:
        batch_x = imgs[batch_idx]
        batch_x = batch_x / 255.0

        batch_y = labels[batch_idx]

        supervised.fit(batch_x, batch_y, batch_size=BATCH_SIZE, epochs=1)

extractor.save('extractor.tf')
