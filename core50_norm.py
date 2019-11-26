import numpy as np


with np.load('core50/features.npz') as core50:
        x = core50['x']
        instance = core50['instance']
        category = core50['category']
        session = core50['session']

x = x / x.std()

np.savez('core50/features_norm.npz', x=x, instance=instance, session=session, category=category)
