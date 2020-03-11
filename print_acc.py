import glob
import pickle
import os
import numpy as np
import math

dirs = ['tc_rep', 'tc_norep', 'notc_rep', 'notc_norep']


def get_mean(profiles, lvl, mem):
    return np.mean([p[f"whole_accuracy_{lvl}_{mem}"][-1] for p in profiles])


def get_std(profiles, lvl, mem):
    return np.std([p[f"whole_accuracy_{lvl}_{mem}"][-1] for p in profiles])


def get_confidence(profiles, lvl, mem):
    return 1.96 * get_std(profiles, lvl, mem) / math.sqrt(len(profiles))


for d in dirs:
    batch = glob.glob(os.path.join('profiling', d, 'profile_0*'))
    ni = glob.glob(os.path.join('profiling', d, 'profile_1*'))
    nc = glob.glob(os.path.join('profiling', d, 'profile_2*'))
    nic = glob.glob(os.path.join('profiling', d, 'profile_3*'))

    scenarios = {
        'batch': [pickle.load(open(f, 'rb')) for f in batch],
        'ni': [pickle.load(open(f, 'rb')) for f in ni],
        'nc': [pickle.load(open(f, 'rb')) for f in nc],
        'nic': [pickle.load(open(f, 'rb')) for f in nic]
    }

    print(d)
    for k, s in scenarios.items():
        print('\t' + k)
        for lvl in ['inst', 'cat']:
            print('\t\t' + lvl)
            for mem in ['episodic', 'semantic']:
                print('\t\t\t' + mem)
                print("\t\t\t\t{0:.3f} +- {1:.3f}".format(get_mean(s, lvl, mem), get_confidence(s, lvl, mem)))
