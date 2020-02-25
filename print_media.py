import glob
import sys
import math
import itertools
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="white")


def reduce_dims(vec, idxs):
    res = []
    for i in idxs:
        res.append(vec[i])
    return res


def print_sns(dicts, key):
    print(key)
    seqs = [d[key] for d in dicts]
    if len(seqs[0]) > 100:
        idxs = range(0, len(seqs[0]), int(len(seqs[0]) / 100))
        seqs = [reduce_dims(s, idxs) for s in seqs]
    seqs = [zip(range(len(s)), s) for s in seqs]
    seqs = itertools.chain(*seqs)
    df = pd.DataFrame(seqs, columns=['time', 'value'])
    df = df.sort_values(by=['time'])
    print(df)
    plot = sns.lineplot(x='time', y='value', ci='sd', data=df, palette="deep")
    plot.figure.savefig(f"media/ni_{key}.png")


if len(sys.argv) < 2:
    print('usage: {} path'.format(sys.argv[0]))
    exit(0)
path = sys.argv[1]

cumuls = glob.glob(path + 'profile_0_*')
ni = glob.glob(path + 'profile_1_*')
nc = glob.glob(path + 'profile_2_*')
nic = glob.glob(path + 'profile_3_*')

assert len(cumuls) > 0, "couldn't find cumulative profiling files"
assert len(ni) > 0, "couldn't find ni profiling files"
assert len(nc) > 0, "couldn't find nc profiling files"
assert len(nic) > 0, "couldn't find nic profiling files"

dicts = []
for file in ni:
    with open(file, 'rb') as f:
        dicts.append(pickle.load(f))

for key in dicts[0].keys():
    print(key)
    seqs = [d[key] for d in dicts]
    if len(seqs[0]) > 100:
        idxs = range(0, len(seqs[0]), int(len(seqs[0]) / 100))
        seqs = [list(reduce_dims(s, idxs)) for s in seqs]
    seqs = [zip(range(len(s)), s) for s in seqs]
    seqs = itertools.chain(*seqs)
    df = pd.DataFrame(seqs, columns=['time', 'value'])
    df = df.sort_values(by=['time'])
    plot = sns.lineplot(x='time', y='value', ci='sd', data=df, palette="deep")
    plot.figure.savefig(f"media/ni_{key}.png")
    plot.figure.clf()

# print_sns(dicts, 'num_nodes_semantic')
