import tensorflow as tf
import numpy as np


SPLIT_SIZE = 100

extractor = tf.keras.models.load_model('extractor2.tf')
extractor.summary()

# load and preprocess images
core50_5fps = np.load('core50/5fps.npz')
imgs = core50_5fps['x']
labels = core50_5fps['y']
num_classes = 50
num_samples = len(labels)
num_batches = int(num_samples / SPLIT_SIZE) + 1

print("num samples: {}".format(num_samples))

features = []

for i in range(num_batches):
    print("{}/{}: {}".format((i + 1) * SPLIT_SIZE, num_samples, (((i + 1) * SPLIT_SIZE) / num_samples) * 100))
    batch_x = imgs[i * SPLIT_SIZE: (i + 1) * SPLIT_SIZE]
    batch_x = batch_x / 255.0

    res = extractor.predict_on_batch(batch_x)
    for r in res:
        features.append(r)

features = np.array(features)
# features = features.reshape(-1, features.shape[-1])
print("final shape: {}".format(features.shape))

np.savez('core50/5fps_features_256.npz', x=features, y=labels)
