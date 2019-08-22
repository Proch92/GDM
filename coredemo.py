# -*- coding: utf-8 -*-
"""
Dual-memory Incremental learning with memory replay
@last-modified: 30 November 2018
@author: German I. Parisi (german.parisi@gmail.com)

"""

import numpy as np
from episodic_gwr import EpisodicGWR


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
            r_weights[i][r] = net.weights[int(samples[r])][0]
            for l in range(0, len(net.num_labels)):
                r_labels[i][l][r] = np.argmax(net.alabels[l][int(samples[r])])
    return r_weights, r_labels


if __name__ == "__main__":

    train_flag = True
    train_type = 1  # 0: Batch, 1: NI, 2: NC, 3: NIC
    train_replay = True

    core50 = np.load('core50/5fps_256.npz')
    core50_x = core50['features']
    core50_instances = core50['instance']
    core50_categories = core50['category']
    core50_sessions = core50['session']
    assert len(np.unique(core50_instances)) == 50, "number of unique instances != 50"
    assert len(np.unique(core50_categories)) == 10, "number of unique categories != 10"
    assert len(np.unique(core50_sessions)) == 11, "number of sessions != 11"
    assert len(core50_x) == len(core50_categories) == len(core50_categories) == len(core50_sessions), "wrong number of samples and labels"

    train_x = [sample for i, sample in enumerate(core50_x) if core50_sessions[i] not in [3, 7, 10]]
    train_instances = [instance for i, instance in enumerate(core50_instances) if core50_sessions[i] not in [3, 7, 10]]
    train_categories = [category for i, category in enumerate(core50_categories) if core50_sessions[i] not in [3, 7, 10]]
    test_x = [sample for i, sample in enumerate(core50_x) if core50_sessions[i] in [3, 7, 10]]
    test_instances = [instance for i, instance in enumerate(core50_instances) if core50_sessions[i] in [3, 7, 10]]
    test_categories = [category for i, category in enumerate(core50_categories) if core50_sessions[i] in [3, 7, 10]]

    assert train_type < 4, "Invalid type of training."

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
    ds_labels = np.zeros((len(e_labels), len(train_instances)))
    ds_labels[0] = train_instances
    ds_labels[1] = train_categories

    num_context = 2  # number of context descriptors
    epochs = 1  # epochs per sample for incremental learning
    a_threshold = [0.3, 0.001]
    beta = 0.7
    learning_rates = [0.5, 0.005]
    context = True

    g_episodic = EpisodicGWR()
    g_episodic.init_network(train_x, e_labels, num_context)

    g_semantic = EpisodicGWR()
    g_semantic.init_network(train_x, s_labels, num_context)

    if train_type == 0:
        # Batch training
        # Train episodic memory
        g_episodic.train_egwr(train_x, ds_labels, 7, a_threshold[0], beta, learning_rates, context, regulated=0)

        e_weights, e_labels = g_episodic.test(train_x, ds_labels, ret_vecs=True)
        # Train semantic memory
        g_semantic.train_egwr(e_weights, e_labels, 7, a_threshold[1], beta, learning_rates, context, regulated=1)
    elif train_type == 1:
        # Incremental training New Instance NI
        n_episodes = 0
        batch_size = 10  # number of samples per epoch
        # Replay parameters
        replay_size = (num_context * 2) + 1  # size of RNATs
        replay_weights = []
        replay_labels = []

        # Train episodic memory
        for s in range(0, train_x.shape[0], batch_size):
            print("batch: ({} - {})".format(s, s + batch_size))
            g_episodic.train_egwr(train_x[s:s + batch_size],
                                  ds_labels[:, s:s + batch_size],
                                  epochs, a_threshold[0], beta, learning_rates,
                                  context, regulated=0)

            e_weights, e_labels = g_episodic.test(train_x, ds_labels,
                                                  ret_vecs=True)
            # Train semantic memory
            g_semantic.train_egwr(e_weights[s:s + batch_size],
                                  e_labels[:, s:s + batch_size],
                                  epochs, a_threshold[1], beta, learning_rates,
                                  context, regulated=1)

            if train_replay and n_episodes > 0:
                # Replay pseudo-samples
                for r in range(0, replay_weights.shape[0]):
                    g_episodic.train_egwr(replay_weights[r], replay_labels[r, :],
                                          epochs, a_threshold[0], beta,
                                          learning_rates, 0, 0)

                    g_semantic.train_egwr(replay_weights[r], replay_labels[r],
                                          epochs, a_threshold[1], beta,
                                          learning_rates, 0, 1)

            # Generate pseudo-samples
            if train_replay:
                replay_weights, replay_labels = replay_samples(g_episodic, replay_size)

            n_episodes += 1

    e_weights, e_labels = g_episodic.test(test_x, ds_labels, test_accuracy=True, ret_vecs=True)
    g_semantic.test(e_weights, e_labels, test_accuracy=True)

    print("Accuracy episodic: %s, semantic: %s" % (g_episodic.test_accuracy[0], g_semantic.test_accuracy[0]))
