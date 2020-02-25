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


def print_sns(dicts, key, close=False, bottom_lim=0.0):
    print(key)
    seqs = [d[key] for d in dicts]
    if len(seqs[0]) > 100:
        idxs = range(0, len(seqs[0]), int(len(seqs[0]) / 100))
        seqs = [reduce_dims(s, idxs) for s in seqs]
    seqs = [zip(range(len(s)), s) for s in seqs]
    seqs = itertools.chain(*seqs)
    df = pd.DataFrame(seqs, columns=['time', 'value'])
    df = df.sort_values(by=['time'])
    plot = sns.lineplot(x='time', y='value', ci='sd', data=df, palette="deep")
    plot.set_ylim(bottom=bottom_lim)
    plot.figure.savefig(f"media/ni_{key}.png")
    if close:
        plot.figure.clf()


def get_dicts(files):
    dicts = []
    for file in files:
        with open(file, 'rb') as f:
            dicts.append(pickle.load(f))
    return dicts


def build_seqs(key, dicts):
    seqs = [d[key] for d in dicts]
    if len(seqs[0]) > 100:
        idxs = range(0, len(seqs[0]), int(len(seqs[0]) / 100))
        seqs = [reduce_dims(s, idxs) for s in seqs]
    return seqs


if len(sys.argv) < 2:
    print('usage: {} path'.format(sys.argv[0]))
    exit(0)
path = sys.argv[1]

# cumuls = glob.glob(path + 'profile_0_*')
# ni = glob.glob(path + 'profile_1_*')
# nc = glob.glob(path + 'profile_2_*')
# nic = glob.glob(path + 'profile_3_*')

# assert len(cumuls) > 0, "couldn't find cumulative profiling files"
# assert len(ni) > 0, "couldn't find ni profiling files"
# assert len(nc) > 0, "couldn't find nc profiling files"
# assert len(nic) > 0, "couldn't find nic profiling files"

# print_sns(dicts, 'num_nodes_semantic')
####################################################################

batch = glob.glob(path + 'profile_0_*')
ni = glob.glob(path + 'profile_1_*')
nc = glob.glob(path + 'profile_2_*')
nic = glob.glob(path + 'profile_3_*')

if len(batch) > 0:
    dicts = get_dicts(batch)
    # neuron number
    seqs = build_seqs("num_nodes_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("num_nodes_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['time', 'neuroni', 'memoria'])
    plot = sns.lineplot(x='time', y='neuroni', ci='sd', data=df, palette="deep", hue="memoria")
    # plot.set_ylim(bottom=0.0)
    plot.figure.savefig(f"media/batch/nneurons.png")
    plot.figure.clf()

    # instance level
    seqs = build_seqs("whole_accuracy_inst_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_inst_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.8, top=1.0)
    plot.figure.savefig(f"media/batch/whole_acc_inst.png")
    plot.figure.clf()

    # category level
    seqs = build_seqs("whole_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.8, top=1.0)
    plot.figure.savefig(f"media/batch/whole_acc_cat.png")
    plot.figure.clf()

# ################################################ NI #########

if len(ni) > 0:
    dicts = get_dicts(ni)
    # neuron number
    seqs = build_seqs("num_nodes_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("num_nodes_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['time', 'neuroni', 'memoria'])
    plot = sns.lineplot(x='time', y='neuroni', ci='sd', data=df, palette="deep", hue="memoria")
    # plot.set_ylim(bottom=0.0)
    plot.figure.savefig(f"media/ni/nneurons.png")
    plot.figure.clf()

    # #################### whole ##
    # instance level
    seqs = build_seqs("whole_accuracy_inst_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_inst_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"media/ni/whole_acc_inst.png")
    plot.figure.clf()

    # category level
    seqs = build_seqs("whole_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"media/ni/whole_acc_cat.png")
    plot.figure.clf()

    # #################### first ##
    # instance level
    seqs = build_seqs("first_accuracy_inst_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("first_accuracy_inst_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"media/ni/first_acc_inst.png")
    plot.figure.clf()

    # category level
    seqs = build_seqs("first_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("first_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"media/ni/first_acc_cat.png")
    plot.figure.clf()

    # #################### seen ##
    # instance level
    seqs = build_seqs("seen_accuracy_inst_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("seen_accuracy_inst_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"media/ni/seen_acc_inst.png")
    plot.figure.clf()

    # category level
    seqs = build_seqs("seen_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("seen_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"media/ni/seen_acc_cat.png")
    plot.figure.clf()

# ################################################ NC #########

if len(nc) > 0:
    dicts = get_dicts(nc)
    # neuron number
    seqs = build_seqs("num_nodes_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("num_nodes_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['time', 'neuroni', 'memoria'])
    plot = sns.lineplot(x='time', y='neuroni', ci='sd', data=df, palette="deep", hue="memoria")
    # plot.set_ylim(bottom=0.0)
    plot.figure.savefig(f"media/nc/nneurons.png")
    plot.figure.clf()

    # #################### whole ##
    # instance level
    seqs = build_seqs("whole_accuracy_inst_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_inst_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.0, top=1.0)
    plot.figure.savefig(f"media/nc/whole_acc_inst.png")
    plot.figure.clf()

    # category level
    seqs = build_seqs("whole_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.0, top=1.0)
    plot.figure.savefig(f"media/nc/whole_acc_cat.png")
    plot.figure.clf()

    # #################### first ##
    # instance level
    seqs = build_seqs("first_accuracy_inst_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("first_accuracy_inst_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.55, top=1.0)
    plot.figure.savefig(f"media/nc/first_acc_inst.png")
    plot.figure.clf()

    # category level
    seqs = build_seqs("first_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("first_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.55, top=1.0)
    plot.figure.savefig(f"media/nc/first_acc_cat.png")
    plot.figure.clf()

    # #################### seen ##
    # instance level
    seqs = build_seqs("seen_accuracy_inst_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("seen_accuracy_inst_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"media/nc/seen_acc_inst.png")
    plot.figure.clf()

    # category level
    seqs = build_seqs("seen_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("seen_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"media/nc/seen_acc_cat.png")
    plot.figure.clf()

# ################################################ NIC #########

if len(nic) > 0:
    dicts = get_dicts(nic)
    # neuron number
    seqs = build_seqs("num_nodes_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("num_nodes_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['time', 'neuroni', 'memoria'])
    plot = sns.lineplot(x='time', y='neuroni', ci='sd', data=df, palette="deep", hue="memoria")
    # plot.set_ylim(bottom=0.0)
    plot.figure.savefig(f"media/nic/nneurons.png")
    plot.figure.clf()

    # #################### whole ##
    # instance level
    seqs = build_seqs("whole_accuracy_inst_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_inst_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.0, top=1.0)
    plot.figure.savefig(f"media/nic/whole_acc_inst.png")
    plot.figure.clf()

    # category level
    seqs = build_seqs("whole_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.0, top=1.0)
    plot.figure.savefig(f"media/nic/whole_acc_cat.png")
    plot.figure.clf()

    # #################### first ##
    # instance level
    seqs = build_seqs("first_accuracy_inst_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("first_accuracy_inst_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.5, top=1.0)
    plot.figure.savefig(f"media/nic/first_acc_inst.png")
    plot.figure.clf()

    # category level
    seqs = build_seqs("first_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("first_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.5, top=1.0)
    plot.figure.savefig(f"media/nic/first_acc_cat.png")
    plot.figure.clf()

    # #################### seen ##
    # instance level
    seqs = build_seqs("seen_accuracy_inst_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("seen_accuracy_inst_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.6, top=1.0)
    plot.figure.savefig(f"media/nic/seen_acc_inst.png")
    plot.figure.clf()

    # category level
    seqs = build_seqs("seen_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("seen_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['epoch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='epoch', y='accuracy', ci='sd', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.6, top=1.0)
    plot.figure.savefig(f"media/nic/seen_acc_cat.png")
    plot.figure.clf()
