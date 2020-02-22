import pickle
import matplotlib.pyplot as plt

import sys

if len(sys.argv) < 3:
    print('usage: {} infile out_file_name'.format(sys.argv[0]))
    exit(0)
file = sys.argv[1]

with open(file, 'rb') as f:
    profile = pickle.load(f)

# num_nodes_episodic
# num_nodes_semantic
# activity_episodic
# activity_semantic
# update_rate_episodic
# update_rate_semantic
# no_instances_seen
# no_categories_seen
# aqe_episodic
# aqe_semantic
# seen_accuracy_inst_episodic
# seen_accuracy_cat_episodic
# seen_accuracy_inst_semantic
# seen_accuracy_cat_semantic
# first_accuracy_inst_episodic
# first_accuracy_cat_episodic
# first_accuracy_inst_semantic
# first_accuracy_cat_semantic
# whole_accuracy_inst_episodic
# whole_accuracy_cat_episodic
# whole_accuracy_inst_semantic
# whole_accuracy_cat_semantic

fig, ax = plt.subplots()

ax.set_title('# neurons')
ax.get_xaxis().set_visible(False)
ax.plot(profile['num_nodes_episodic'], label='episodic')
ax.plot(profile['num_nodes_semantic'], label='semantic')

plt.savefig('media/nneurons' + sys.argv[2] + '.png')


fig, axs = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.4)

axs[0].set_title('update rate')
axs[0].get_xaxis().set_visible(False)
axs[0].plot(profile['update_rate_episodic'], label='episodic')
axs[0].plot(profile['update_rate_semantic'], label='semantic')

axs[1].set_title('AQE (loss)')
axs[1].plot(profile['aqe_episodic'], label='episodic')
axs[1].plot(profile['aqe_semantic'], label='semantic')

axs[0].legend()
axs[1].legend()

plt.savefig('media/' + sys.argv[2] + '.png')


if 'seen_accuracy_inst_episodic' in profile:
    # fig, axs = plt.subplots(3, 1)
    # fig.subplots_adjust(hspace=0.5)

    fig, ax = plt.subplots()
    ax.set_title('test acc batches seen')
    ax.plot(profile['seen_accuracy_inst_episodic'], label='inst epis')
    ax.plot(profile['seen_accuracy_cat_episodic'], label='cat epis')
    ax.plot(profile['seen_accuracy_inst_semantic'], label='inst sem')
    ax.plot(profile['seen_accuracy_cat_semantic'], label='cat sem')
    ax.legend()

    plt.savefig('media/accuracies_batchseen' + sys.argv[2] + '.png')

    fig, ax = plt.subplots()
    ax.set_title('test acc first batch')
    ax.plot(profile['first_accuracy_inst_episodic'], label='inst epis')
    ax.plot(profile['first_accuracy_cat_episodic'], label='cat epis')
    ax.plot(profile['first_accuracy_inst_semantic'], label='inst sem')
    ax.plot(profile['first_accuracy_cat_semantic'], label='cat sem')
    ax.legend()

    plt.savefig('media/accuracies_firstbatch' + sys.argv[2] + '.png')

    fig, ax = plt.subplots()
    ax.set_title('test acc')
    ax.plot(profile['whole_accuracy_inst_episodic'], label='inst epis')
    ax.plot(profile['whole_accuracy_cat_episodic'], label='cat epis')
    ax.plot(profile['whole_accuracy_inst_semantic'], label='inst sem')
    ax.plot(profile['whole_accuracy_cat_semantic'], label='cat sem')
    ax.legend()

    plt.savefig('media/accuracies_wholetest' + sys.argv[2] + '.png')
