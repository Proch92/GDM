"""
gwr-tb :: Episodic-GWR
@last-modified: 25 January 2019
@author: German I. Parisi (german.parisi@gmail.com)

"""

import numpy as np
import math
from gammagwr import GammaGWR
import publish


class EpisodicGWR(GammaGWR):

    def __init__(self, name):
        super().__init__(name)
        self.iterations = 0

    def init_network(self, ds, e_labels, num_context) -> None:

        assert self.iterations < 1, "Can't initialize a trained network"
        assert ds is not None, "Need a dataset to initialize a network"

        # Lock to prevent training
        self.locked = False

        # Start with 2 neurons
        self.num_nodes = 2
        self.dimension = ds.shape[1]
        self.num_context = num_context
        self.depth = self.num_context + 1
        empty_neuron = np.zeros((self.depth, self.dimension))
        self.weights = [empty_neuron, empty_neuron]

        # Global context
        self.g_context = np.zeros((self.depth, self.dimension))

        # Create habituation counters
        self.habn = [1, 1]

        # Create edge and age matrices
        self.edges = np.ones((self.num_nodes, self.num_nodes))
        self.ages = np.zeros((self.num_nodes, self.num_nodes))

        # new --------------------------------
        # Temporal connections
        self.temporal = np.zeros((self.num_nodes, self.num_nodes))

        # Label histogram
        self.num_labels = e_labels
        self.alabels = []
        for l in range(0, len(self.num_labels)):
            self.alabels.append(-np.ones((self.num_nodes, self.num_labels[l])))
        init_ind = list(range(0, self.num_nodes))
        for i in range(0, len(init_ind)):
            self.weights[i][0] = ds[i]

        # Context coefficients
        self.alphas = self.compute_alphas(self.depth)
        self.alphas_T = self.alphas.T

    # par 3.2 episodic memory
    # hebbian update temporal synaptic link
    # empowering link from previous_ind (t-1) to current_ind (t)
    # used to produce temporal trajectories
    def update_temporal(self, current_ind, previous_ind, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)
        if new_node:
            self.temporal = super().expand_matrix(self.temporal)
        if previous_ind != -1 and previous_ind != current_ind:
            self.temporal[previous_ind, current_ind] += 1

    def update_labels(self, bmu, label, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)
        if not new_node:
            for l in range(0, len(self.num_labels)):
                for a in range(0, self.num_labels[l]):
                    if a == label[l]:
                        self.alabels[l][bmu, a] += self.a_inc
                    else:
                        if label[l] != -1:
                            self.alabels[l][bmu, a] -= self.a_dec
                            if (self.alabels[l][bmu, a] < 0):
                                self.alabels[l][bmu, a] = 0
        else:
            for l in range(0, len(self.num_labels)):
                new_alabel = np.zeros((1, self.num_labels[l]))
                if label[l] != -1:
                    new_alabel[0, int(label[l])] = self.a_inc
                self.alabels[l] = np.concatenate(
                    (self.alabels[l], new_alabel), axis=0)

    def remove_isolated_nodes(self) -> None:
        cnt_deleted = 0
        if self.num_nodes > 2:
            neigh_sum = np.add.reduce(self.edges)
            to_delete = np.where(neigh_sum.flatten() == 0)[0]
            cnt_deleted = len(to_delete)

            if self.num_nodes - cnt_deleted < 2:
                cnt_deleted = self.num_nodes - 2
                to_delete = to_delete[:cnt_deleted]

            self.weights = [w for i, w in enumerate(self.weights) if i not in to_delete]
            self.habn = [w for i, w in enumerate(self.habn) if i not in to_delete]
            self.alabels = [np.delete(lab, to_delete, axis=0) for lab in self.alabels]
            self.edges = np.delete(self.edges, to_delete, axis=0)
            self.edges = np.delete(self.edges, to_delete, axis=1)
            self.ages = np.delete(self.ages, to_delete, axis=0)
            self.ages = np.delete(self.ages, to_delete, axis=1)
            self.temporal = np.delete(self.temporal, to_delete, axis=0)
            self.temporal = np.delete(self.temporal, to_delete, axis=1)
            self.num_nodes = self.num_nodes - cnt_deleted

        # print("(-- Removed %s neuron(s))" % cnt_deleted)

    def train_egwr(self, ds_vectors, ds_labels, epochs, a_threshold,
                   context, parameters, regulated) -> None:

        assert not self.locked, "Network is locked. Unlock to train."

        self.samples = ds_vectors.shape[0]
        self.max_epochs = epochs
        self.a_threshold = a_threshold
        self.epsilon_b = parameters["learning_rate_bmu"]
        self.epsilon_n = parameters["learning_rate_neighbours"]
        self.beta = parameters["beta"]
        self.regulated = regulated
        self.context = context
        if not self.context:
            self.g_context.fill(0)
        self.hab_threshold = parameters["habituation_threshold"]
        self.tau_b = parameters["tau_bmu"]
        self.tau_n = parameters["tau_neighbours"]
        if parameters["max_nodes"] == 0:
            self.max_nodes = self.samples  # OK for batch, bad for incremental
        else:
            self.max_nodes = parameters["max_nodes"]
        self.max_neighbors = parameters["max_neighbours"]
        self.max_age = parameters["max_age"]
        self.new_node = parameters["new_node"]
        self.a_inc = parameters["labels_a_inc"]
        self.a_dec = parameters["labels_a_dec"]
        self.mod_rate = parameters["mod_rate"]

        # Start training
        error_counter = np.zeros(self.max_epochs)
        previous_bmu = np.zeros((self.depth, self.dimension))
        previous_ind = -1
        for epoch in range(self.max_epochs):
            for iteration in range(self.samples):
                # if self.iterations % 100 == 0:
                #    print('epoch: {} / {} - {:.2f}%'.format(epoch + 1, self.max_epochs, (iteration / self.samples) * 100))

                # Generate input sample
                self.g_context[0] = ds_vectors[iteration]
                label = ds_labels[:, iteration]

                # Update global context
                for z in range(1, self.depth):
                    self.g_context[z] = (
                        self.beta * previous_bmu[z]) + ((1 - self.beta) * previous_bmu[z - 1])

                # Find the best and second-best matching neurons
                b_index, b_distance, s_index = super().find_bmus(self.g_context, s_best=True)

                b_label = np.argmax(self.alabels[0][b_index])
                misclassified = b_label != label[0]

                # Quantization error
                error_counter[epoch] += b_distance

                # Compute network activity
                a = math.exp(-b_distance)
                publish.send('activity_' + self.name, a)

                publish.send('update_rate_' + self.name, np.mean(self.habn) * self.epsilon_b)

                # Store BMU at time t for t+1
                previous_bmu = self.weights[b_index]

                if (not self.regulated) or (self.regulated and misclassified):

                    if (a < self.a_threshold and
                            self.habn[b_index] < self.hab_threshold and
                            self.num_nodes < self.max_nodes):
                        # Add new neuron
                        n_index = self.num_nodes
                        super().add_node(b_index)

                        # Add label histogram
                        self.update_labels(n_index, label, new_node=True)

                        # Update edges and ages
                        super().update_edges(b_index, s_index, new_index=n_index)

                        # Update temporal connections
                        self.update_temporal(
                            n_index, previous_ind, new_node=True)

                        # Habituation counter
                        super().habituate_node(n_index, self.tau_b, new_node=True)

                    else:
                        # Habituate BMU
                        super().habituate_node(b_index, self.tau_b)

                        # Update BMU's weight vector
                        b_rate, n_rate = self.epsilon_b, self.epsilon_n
                        if self.regulated and misclassified:
                            b_rate *= self.mod_rate
                            n_rate *= self.mod_rate
                        else:
                            # Update BMU's label histogram
                            self.update_labels(b_index, label)

                        super().update_weight(b_index, b_rate)

                        # Update BMU's edges // Remove BMU's oldest ones
                        super().update_edges(b_index, s_index)

                        # Update temporal connections
                        self.update_temporal(b_index, previous_ind)

                        # Update BMU's neighbors
                        super().update_neighbors(b_index, n_rate)

                self.iterations += 1

                previous_ind = b_index

                publish.send('num_nodes_' + self.name, self.num_nodes)

            # Remove old edges
            super().remove_old_edges()

            # Average quantization error (AQE)
            error_counter[epoch] /= self.samples
            publish.send('aqe_' + self.name, error_counter[epoch])

            # print("(Epoch: %s, NN: %s, ATQE: %s)" %
            #      (epoch + 1, self.num_nodes, error_counter[epoch]))

        # Remove isolated neurons
        self.remove_isolated_nodes()

        return error_counter

    def test(self, ds_vectors, ds_labels, **kwargs):
        test_accuracy = kwargs.get('test_accuracy', False)
        test_vecs = kwargs.get('ret_vecs', False)
        test_samples = ds_vectors.shape[0]
        bmus_index = -np.ones(test_samples)
        bmus_weight = np.zeros((test_samples, self.dimension))
        bmus_label = -np.ones((len(self.num_labels), test_samples))
        bmus_activation = np.zeros(test_samples)

        input_context = np.zeros((self.depth, self.dimension))

        if test_accuracy:
            acc_counter = np.zeros(len(self.num_labels))

        for i in range(0, test_samples):
            input_context[0] = ds_vectors[i]
            # Find the BMU
            b_index, b_distance = super().find_bmus(input_context)
            bmus_index[i] = b_index
            bmus_weight[i] = self.weights[b_index][0]
            bmus_activation[i] = math.exp(-b_distance)
            for l in range(0, len(self.num_labels)):
                bmus_label[l, i] = np.argmax(self.alabels[l][b_index])

            for j in range(1, self.depth):
                input_context[j] = input_context[j - 1]

            if test_accuracy:
                for l in range(0, len(self.num_labels)):
                    if bmus_label[l, i] == ds_labels[l, i]:
                        acc_counter[l] += 1

        if test_accuracy:
            self.test_accuracy = acc_counter / test_samples

        if test_vecs:
            s_labels = -np.ones((len(self.num_labels), test_samples))
            for l in range(len(self.num_labels)):
                s_labels[l] = bmus_label[l]
            return bmus_weight, s_labels
