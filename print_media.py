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


def get_dicts(files):
    dicts = []
    for file in files:
        with open(file, 'rb') as f:
            dicts.append(pickle.load(f))
    return dicts


def clip(vec, length=100):
    if len(vec) > 100:
        idxs = range(0, len(vec), int(len(vec) / length))
        return reduce_dims(vec, idxs)
    return vec


def build_seqs(key, dicts):
    print(key)
    seqs = [d[key] for d in dicts]
    seqs = [clip(s) for s in seqs]
    return seqs


if len(sys.argv) < 3:
    print('usage: {} path out_path'.format(sys.argv[0]))
    exit(0)
path = sys.argv[1]
out_path = sys.argv[2]
out_path = out_path.strip('/')

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
    plot = sns.lineplot(x='time', y='neuroni', data=df, palette="deep", hue="memoria")
    # plot.set_ylim(bottom=0.0)
    plot.figure.savefig(f"{out_path}/batch/nneurons.png", dpi=300)
    plot.figure.clf()

    # instance level
    seqs = build_seqs("whole_accuracy_inst_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_inst_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.8, top=1.0)
    plot.figure.savefig(f"{out_path}/batch/whole_acc_inst.png", dpi=300)
    plot.figure.clf()

    # category level
    seqs = build_seqs("whole_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.8, top=1.0)
    plot.figure.savefig(f"{out_path}/batch/whole_acc_cat.png", dpi=300)
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
    plot = sns.lineplot(x='time', y='neuroni', data=df, palette="deep", hue="memoria")
    # plot.set_ylim(bottom=0.0)
    plot.figure.savefig(f"{out_path}/ni/nneurons.png", dpi=300)
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
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"{out_path}/ni/whole_acc_inst.png", dpi=300)
    plot.figure.clf()

    # category level
    seqs = build_seqs("whole_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"{out_path}/ni/whole_acc_cat.png", dpi=300)
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
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"{out_path}/ni/first_acc_inst.png", dpi=300)
    plot.figure.clf()

    # category level
    seqs = build_seqs("first_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("first_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"{out_path}/ni/first_acc_cat.png", dpi=300)
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
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"{out_path}/ni/seen_acc_inst.png", dpi=300)
    plot.figure.clf()

    # category level
    seqs = build_seqs("seen_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("seen_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"{out_path}/ni/seen_acc_cat.png", dpi=300)
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
    plot = sns.lineplot(x='time', y='neuroni', data=df, palette="deep", hue="memoria")
    # plot.set_ylim(bottom=0.0)
    plot.figure.savefig(f"{out_path}/nc/nneurons.png", dpi=300)
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
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.0, top=1.0)
    plot.figure.savefig(f"{out_path}/nc/whole_acc_inst.png", dpi=300)
    plot.figure.clf()

    # category level
    seqs = build_seqs("whole_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.0, top=1.0)
    plot.figure.savefig(f"{out_path}/nc/whole_acc_cat.png", dpi=300)
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
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.55, top=1.0)
    plot.figure.savefig(f"{out_path}/nc/first_acc_inst.png", dpi=300)
    plot.figure.clf()

    # category level
    seqs = build_seqs("first_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("first_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.55, top=1.0)
    plot.figure.savefig(f"{out_path}/nc/first_acc_cat.png", dpi=300)
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
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"{out_path}/nc/seen_acc_inst.png", dpi=300)
    plot.figure.clf()

    # category level
    seqs = build_seqs("seen_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("seen_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.7, top=1.0)
    plot.figure.savefig(f"{out_path}/nc/seen_acc_cat.png", dpi=300)
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
    plot = sns.lineplot(x='time', y='neuroni', data=df, palette="deep", hue="memoria")
    # plot.set_ylim(bottom=0.0)
    plot.figure.savefig(f"{out_path}/nic/nneurons.png", dpi=300)
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
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.0, top=1.0)
    plot.figure.savefig(f"{out_path}/nic/whole_acc_inst.png", dpi=300)
    plot.figure.clf()

    # category level
    seqs = build_seqs("whole_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("whole_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.0, top=1.0)
    plot.figure.savefig(f"{out_path}/nic/whole_acc_cat.png", dpi=300)
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
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.5, top=1.0)
    plot.figure.savefig(f"{out_path}/nic/first_acc_inst.png", dpi=300)
    plot.figure.clf()

    # category level
    seqs = build_seqs("first_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("first_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.5, top=1.0)
    plot.figure.savefig(f"{out_path}/nic/first_acc_cat.png", dpi=300)
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
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.6, top=1.0)
    plot.figure.savefig(f"{out_path}/nic/seen_acc_inst.png", dpi=300)
    plot.figure.clf()

    # category level
    seqs = build_seqs("seen_accuracy_cat_episodic", dicts)
    labels = ["episodica"] * len(seqs[0])
    seqs_episodic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = build_seqs("seen_accuracy_cat_semantic", dicts)
    labels = ["semantica"] * len(seqs[0])
    seqs_semantic = [zip(range(len(s)), s, labels) for s in seqs]
    seqs = itertools.chain(*seqs_episodic, *seqs_semantic)
    df = pd.DataFrame(seqs, columns=['batch', 'accuracy', 'memoria'])
    plot = sns.lineplot(x='batch', y='accuracy', data=df, palette="deep", hue="memoria")
    plot.set_ylim(bottom=0.6, top=1.0)
    plot.figure.savefig(f"{out_path}/nic/seen_acc_cat.png", dpi=300)
    plot.figure.clf()
