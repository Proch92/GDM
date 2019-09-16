import numpy as np
import pickle as pkl
import re


pkl_file = open('core50/paths.pkl', 'rb')
paths = pkl.load(pkl_file)
imgs = np.load('core50/core50_imgs.npz')['x']

paths = paths[::4]
imgs = imgs[::4]

instance = [int(re.search('/o(.+?)/', path).group(1)) - 1 for path in paths]
session = [int(re.search('s(.+?)/', path).group(1)) for path in paths]
category = [int(i / 5.0) for i in instance]

with open('core50/core50_imgs_5fps.npz', 'wb') as f:
    np.savez(f, x=imgs, instance=instance, session=session, category=category)
