import tensorflow as tf
import numpy as np
import pickle as pkl
import re

# prepare labels
pkl_file = open('core50/paths.pkl', 'rb')
paths = pkl.load(pkl_file)
labels = [int(re.search('/o(.+?)/', path).group(1)) for path in paths]
num_classes = len(np.unique(labels))
print(num_classes)
# onehot encoding
labels = [np.eye(num_classes)[l - 1] for l in labels]
labels = labels[::4]  # 20fps -> 5 fps

# load and preprocess images
imgs = np.load('core50/core50_imgs.npz')['x']
imgs = imgs[::4]  # 20 fps -> 5 fps

with open('5fps.npz', 'wb') as f:
    np.savez(f, x=imgs, y=labels)
