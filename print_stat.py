import pickle
import matplotlib.pyplot as plt

with open('profiling.pkl', 'rb') as f:
    profiling = pickle.load(f)

fig, axs = plt.subplots(3, 1, sharex=True)
fig.subplots_adjust(hspace=0.1)

axs[0].plot(profiling['episode'], profiling['episodic']['no_neurons'])
axs[1].plot(profiling['episode'], profiling['semantic']['no_neurons'])

plt.show()
