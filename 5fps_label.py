import numpy as np
import pickle as pkl
import re

# prepare labels
pkl_file = open('core50/paths.pkl', 'rb')
paths = pkl.load(pkl_file)
paths = paths[::4]

instance = [int(re.search('/o(.+?)/', path).group(1)) - 1 for path in paths]
session = [int(re.search('s(.+?)/', path).group(1)) for path in paths]
category = [int((i - 1) / 5.0) for i in instance]


# load and preprocess images
core50 = np.load('core50/5fps_features_256.npz')

with open('core50/5fps_256.npz', 'wb') as f:
    np.savez(f, features=core50['x'], onehot_y=core50['y'], instance=instance, session=session, category=category)
