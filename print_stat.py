import pickle
import matplotlib.pyplot as plt
import numpy as np

import sys

file = sys.argv[1]

with open(file, 'rb') as f:
    profiling = pickle.load(f)

topics = [
    'episode',
    'num_nodes_episodic',
    'num_nodes_semantic',
    'activity_episodic',
    'activity_semantic',
    'update_rate_episodic',
    'update_rate_semantic',
    'no_instances_seen',
    'no_categories_seen',
    'aqe_episodic',
    'aqe_semantic',
    'seen_accuracy_inst_episodic',
    'seen_accuracy_cat_episodic',
    'seen_accuracy_inst_semantic',
    'seen_accuracy_cat_semantic',
    'first_accuracy_inst_episodic',
    'first_accuracy_cat_episodic',
    'first_accuracy_inst_semantic',
    'first_accuracy_cat_semantic',
    'whole_accuracy_inst_episodic',
    'whole_accuracy_cat_episodic',
    'whole_accuracy_inst_semantic',
    'whole_accuracy_cat_semantic'
]

fig, axs = plt.subplots(3, 1, sharex=True)
fig.subplots_adjust(hspace=0.3)

axs[0].set_title('# neurons')
axs[0].plot(profiling['episode'], profiling['episodic']['no_neurons'], label='episodic')
axs[0].plot(profiling['episode'], profiling['semantic']['no_neurons'], label='semantic')
axs[1].set_title('# instances and categories seen')
axs[1].plot(profiling['episode'], profiling['no_instances_seen'], label='instances')
axs[1].plot(profiling['episode'], profiling['no_categories_seen'], label='categories')
profiling['episodic']['aqe'] = [np.mean(a) for a in profiling['episodic']['aqe']]
profiling['semantic']['aqe'] = [np.mean(a) for a in profiling['semantic']['aqe']]
axs[2].set_title('AQE (loss)')
axs[2].plot(profiling['episode'], profiling['episodic']['aqe'], label='episodic')
axs[2].plot(profiling['episode'], profiling['semantic']['aqe'], label='semantic')


axs[0].legend()
axs[1].legend()
axs[2].legend()

plt.xlabel('episode / batch')
plt.savefig('media/training_stats' + ttype + '.png')

fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.3)

axs[0][0].set_title('Test acc instances seen')
axs[0][0].plot(profiling['episode'], profiling['semantic']['test_accuracy_inst'])
axs[0][1].set_title('Test acc categories seen')
axs[0][1].plot(profiling['episode'], profiling['semantic']['test_accuracy_cat'])
axs[1][0].set_title('Test acc instances first batch')
axs[1][0].plot(profiling['episode'], profiling['semantic']['first_accuracy_inst'])
axs[1][1].set_title('Test acc categories first batch')
axs[1][1].plot(profiling['episode'], profiling['semantic']['first_accuracy_cat'])

plt.xlabel('episode / batch')
plt.savefig('media/accuracies' + ttype + '.png')

fig, axs = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.3)
axs[0].set_title('test acc full testset')
axs[0].plot(profiling['episode'], profiling['episodic']['whole_accuracy_inst'], label='episodic')
axs[1].plot(profiling['episode'], profiling['semantic']['whole_accuracy_inst'], label='semantic')

axs[0].legend()
axs[1].legend()

plt.xlabel('episode / batch')
plt.savefig('media/test_acc' + ttype + '.png')
