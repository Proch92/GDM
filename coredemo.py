# -*- coding: utf-8 -*-
"""
Dual-memory Incremental learning with memory replay
@last-modified: 30 November 2018
@author: German I. Parisi (german.parisi@gmail.com)

"""

import numpy as np
import pandas as pd
import random
import sys
import pickle
from episodic_gwr import EpisodicGWR
from tqdm import tqdm
import config
import rtplot


PROFILING = True


def replay_samples(net, size) -> (np.ndarray, np.ndarray):
    samples = np.zeros(size)
    r_weights = np.zeros((net.num_nodes, size, net.dimension))
    r_labels = np.zeros((net.num_nodes, len(net.num_labels), size))
    for i in range(0, net.num_nodes):
        for r in range(0, size):
            if r == 0:
                samples[r] = i
            else:
                samples[r] = np.argmax(net.temporal[int(samples[r - 1]), :])
            r_weights[i, r] = net.weights[int(samples[r])][0]
            for l in range(0, len(net.num_labels)):
                r_labels[i, l, r] = np.argmax(net.alabels[l][int(samples[r])])
    return r_weights, r_labels


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python ' + sys.argv[0] + ' train_type')
        print('train_type -> 0: Batch, 1: NI, 2: NC, 3: NIC')
        exit(0)

    train_flag = True
    train_type = int(sys.argv[1])  # 0: Batch, 1: NI, 2: NC, 3: NIC

    with np.load('core50/features.npz') as core50:
        core50_x = core50['x']
        core50_instances = core50['instance']
        core50_categories = core50['category']
        core50_categories = core50_categories - 1  # 0 indexing
        core50_sessions = core50['session']

    assert train_type < 4, "Invalid type of training."
    assert np.all(core50_instances < 50), "there are instances > 49"
    assert np.all(core50_instances >= 0), "there are instances < 0"
    assert np.all(core50_categories < 10), "there are categories > 9"
    assert np.all(core50_categories >= 0), "there are categories < 0"
    assert len(np.unique(core50_instances)
               ) == 50, "number of unique instances != 50"
    assert len(np.unique(core50_categories)
               ) == 10, "number of unique categories != 10"
    assert len(np.unique(core50_sessions)) == 11, "number of sessions != 11"
    assert len(core50_x) == len(core50_categories) == len(core50_categories) == len(
        core50_sessions), "wrong number of samples and labels"

    ds = pd.DataFrame({
        'x': range(len(core50_x)),
        'instance': core50_instances,
        'category': core50_categories,
        'session': core50_sessions
    })

    test = ds[ds['session'].isin([3, 7, 10])]
    train = ds[ds['session'].isin([1, 2, 4, 5, 6, 8, 9, 11])]

    '''
    Episodic-GWR supports multi-class neurons.
    Set the number of label classes per neuron and possible labels per class
    e.g. e_labels = [50, 10]
    is two labels per neuron, one with 50 and the other with 10 classes.
    Setting the n. of classes is done for experimental control but it is not
    necessary for associative GWR learning.
    '''
    e_labels = [50, 10]
    s_labels = [50, 10]

    parameters = config.from_file("configs/custom.json")

    context = True
    num_context = parameters["context_depth"]
    epochs = 1  # epochs per sample for incremental learning
    a_threshold = [parameters["insertion_threshold_episodic"], parameters["insertion_threshold_semantic"]]

    g_episodic = EpisodicGWR('episodic')
    g_episodic.init_network(core50_x[train['x'].values], e_labels, num_context)

    g_semantic = EpisodicGWR('semantic')
    g_semantic.init_network(core50_x[train['x'].values], s_labels, num_context)

    # profiling
    instances_seen = set()
    categories_seen = set()
    sessions_seen = set()
    profiling = {
        'episode': [],
        'no_instances_seen': [],
        'no_categories_seen': [],
        'episodic': {
            'no_neurons': [],
            'aqe': [],
            'test_accuracy_inst': [],
            'test_accuracy_cat': [],
            'first_accuracy_inst': [],
            'first_accuracy_cat': [],
            'whole_accuracy_inst': [],
            'whole_accuracy_cat': []
        },
        'semantic': {
            'no_neurons': [],
            'aqe': [],
            'test_accuracy_inst': [],
            'test_accuracy_cat': [],
            'first_accuracy_inst': [],
            'first_accuracy_cat': [],
            'whole_accuracy_inst': [],
            'whole_accuracy_cat': []
        }
    }

    rtplot.plot(topic="num_nodes_episodic", refresh_rate=0.5)
    rtplot.plot(topic="num_nodes_semantic", refresh_rate=0.01)

    if train_type == 0:
        # Batch training
        # Train episodic memory
        x = core50_x[train['x'].values]
        ds_labels = np.zeros((len(e_labels), len(train)))
        ds_labels[0] = train['instance'].values
        ds_labels[1] = train['category'].values

        g_episodic.train_egwr(
            x, ds_labels, epochs, a_threshold[0], context, parameters, regulated=0)

        e_weights, eval_labels = g_episodic.test(x, ds_labels, ret_vecs=True)

        # Train semantic memory
        g_semantic.train_egwr(e_weights, eval_labels, epochs,
                              a_threshold[1], context, parameters, regulated=1)

    else:
        # Incremental training New Instance NI
        batch_size = 10  # number of samples per epoch
        # Replay parameters
        train_replay = False
        replay_size = (num_context * 2) + 1  # size of RNATs
        replay_weights = []
        replay_labels = []

        n_episodes = 0

        # prepare batches
        if train_type == 1:  # NI
            batches = train.groupby('session')
            batches = [batch for _, batch in batches]
        elif train_type == 2:  # NC
            batches = train.groupby('category')
            batches = [batch for _, batch in batches]
            batches[0] = pd.concat([batches[0], batches[1]])
            del batches[1]
        elif train_type == 3:  # NIC
            batches = train.groupby('instance')
            batches = [batch for _, batch in batches]
            random.shuffle(batches)
            while(batches[0]['category'].values[0] == batches[1]['category'].values[0]):
                random.shuffle(batches)
            batches[0] = pd.concat([batches[0], batches[1]])
            del batches[1]

        print('printing batches lens')
        for b in batches:
            print(len(b['instance'].values))

        # Train episodic memory
        for batch in tqdm(batches, desc='Batches'):
            # prepare labels
            ds_labels = np.zeros((len(e_labels), len(batch)))
            ds_labels[0] = batch['instance'].values
            ds_labels[1] = batch['category'].values

            episodic_aqe = g_episodic.train_egwr(core50_x[batch['x'].values],
                                                 ds_labels,
                                                 epochs, a_threshold[0],
                                                 context, parameters, regulated=0)

            e_weights, eval_labels = g_episodic.test(
                core50_x[batch['x'].values], ds_labels, ret_vecs=True)

            assert len(core50_x[batch['x'].values]) == len(e_weights), "test doing crap"

            # Train semantic memory
            semantic_aqe = g_semantic.train_egwr(e_weights,
                                                 eval_labels,
                                                 epochs, a_threshold[1],
                                                 context, parameters, regulated=1)

            if train_replay and n_episodes > 0:
                # Replay pseudo-samples
                for r in range(0, replay_weights.shape[0]):
                    g_episodic.train_egwr(replay_weights[r], replay_labels[r, :],
                                          epochs, a_threshold[0],
                                          0, parameters, regulated=0)

                    g_semantic.train_egwr(replay_weights[r], replay_labels[r],
                                          epochs, a_threshold[1],
                                          0, parameters, regulated=1)

            # Generate pseudo-samples
            if train_replay:
                replay_weights, replay_labels = replay_samples(
                    g_episodic, replay_size)

            # profiling
            if PROFILING:
                profiling['episode'].append(n_episodes)
                for i in batch['instance'].unique():
                    instances_seen.add(i)
                for c in batch['category'].unique():
                    categories_seen.add(c)
                for s in batch['session'].unique():
                    sessions_seen.add(s)
                profiling['no_instances_seen'].append(len(instances_seen))
                profiling['no_categories_seen'].append(len(categories_seen))
                profiling['episodic']['no_neurons'].append(g_episodic.num_nodes)
                profiling['semantic']['no_neurons'].append(g_semantic.num_nodes)
                profiling['episodic']['aqe'].append(episodic_aqe)
                profiling['semantic']['aqe'].append(semantic_aqe)
                if train_type == 1:  # NI
                    test_batch = train[train['session'].isin(sessions_seen)]
                elif train_type == 2:  # NC
                    test_batch = train[train['category'].isin(categories_seen)]
                elif train_type == 3:  # NIC
                    test_batch = train[train['instance'].isin(instances_seen)]
                ds_labels = np.zeros((len(e_labels), len(test_batch)))
                ds_labels[0] = test_batch['instance'].values
                ds_labels[1] = test_batch['category'].values
                e_weights, eval_labels = g_episodic.test(
                    core50_x[test_batch['x'].values], ds_labels, test_accuracy=True, ret_vecs=True)
                g_semantic.test(e_weights, ds_labels, test_accuracy=True)
                profiling['episodic']['test_accuracy_inst'].append(g_episodic.test_accuracy[0])
                profiling['episodic']['test_accuracy_cat'].append(g_episodic.test_accuracy[1])
                profiling['semantic']['test_accuracy_inst'].append(g_semantic.test_accuracy[0])
                profiling['semantic']['test_accuracy_cat'].append(g_semantic.test_accuracy[1])
                ds_labels = np.zeros((len(e_labels), len(batches[0])))
                ds_labels[0] = batches[0]['instance'].values
                ds_labels[1] = batches[0]['category'].values
                e_weights, eval_labels = g_episodic.test(
                    core50_x[batches[0]['x'].values], ds_labels, test_accuracy=True, ret_vecs=True)
                g_semantic.test(e_weights, ds_labels, test_accuracy=True)
                profiling['episodic']['first_accuracy_inst'].append(g_episodic.test_accuracy[0])
                profiling['episodic']['first_accuracy_cat'].append(g_episodic.test_accuracy[1])
                profiling['semantic']['first_accuracy_inst'].append(g_semantic.test_accuracy[0])
                profiling['semantic']['first_accuracy_cat'].append(g_semantic.test_accuracy[1])
                ds_labels = np.zeros((len(e_labels), len(test)))
                ds_labels[0] = test['instance'].values
                ds_labels[1] = test['category'].values
                e_weights, eval_labels = g_episodic.test(
                    core50_x[test['x'].values], ds_labels, test_accuracy=True, ret_vecs=True)
                g_semantic.test(e_weights, ds_labels, test_accuracy=True)
                profiling['episodic']['whole_accuracy_inst'].append(g_episodic.test_accuracy[0])
                profiling['episodic']['whole_accuracy_cat'].append(g_episodic.test_accuracy[1])
                profiling['semantic']['whole_accuracy_inst'].append(g_semantic.test_accuracy[0])
                profiling['semantic']['whole_accuracy_cat'].append(g_semantic.test_accuracy[1])

            n_episodes += 1

    # TEST
    ds_labels = np.zeros((len(e_labels), len(test)))
    ds_labels[0] = test['instance'].values
    ds_labels[1] = test['category'].values
    e_weights, eval_labels = g_episodic.test(
        core50_x[test['x'].values], ds_labels, test_accuracy=True, ret_vecs=True)
    g_semantic.test(e_weights, ds_labels, test_accuracy=True)

    print("Accuracy instance episodic: %s, semantic: %s" %
          (g_episodic.test_accuracy[0], g_semantic.test_accuracy[0]))
    print("Accuracy category episodic: %s, semantic: %s" %
          (g_episodic.test_accuracy[1], g_semantic.test_accuracy[1]))

    if PROFILING:
        with open('profiling/profiling' + str(train_type) + '.pkl', 'wb') as f:
            pickle.dump(profiling, f)
