import numpy as np
# import tensorflow as tf
import random
import sklearn
from sklearn import svm
from sklearn.cluster import KMeans
import sys


with np.load(sys.argv[1]) as core50:
    core50_x = core50['x']
    core50_instances = core50['instance']
    core50_sessions = core50['session']

train_idx = [i for i, s in enumerate(core50_sessions) if s not in [3, 7, 10]]
train_x = core50_x[train_idx]
train_inst = core50_instances[train_idx]

print(train_inst[0:10])

train = list(zip(train_x, train_inst))
random.shuffle(train)
train_x, train_inst = zip(*train)
train_x = np.array(train_x)
tarin_inst = np.array(train_inst)

test_idx = [i for i, s in enumerate(core50_sessions) if s in [3, 7, 10]]
test_x = core50_x[test_idx]
test_inst = core50_instances[test_idx]

# if False:
#     classificator = tf.keras.Sequential([
#         tf.keras.layers.Dropout(0.2, input_shape=(256,)),
#         tf.keras.layers.Dense(50, activation='softmax')
#     ])

#     classificator.compile(
#         optimizer=tf.keras.optimizers.RMSprop(),
#         loss='categorical_crossentropy',
#         metrics=['accuracy'])

#     train_inst_onehot = tf.keras.utils.to_categorical(train_inst, num_classes=50)

#     classificator.fit(train_x, train_inst_onehot, batch_size=32, epochs=3)

#     test_inst_onehot = tf.keras.utils.to_categorical(test_inst, num_classes=50)

#     acc = classificator.test_on_batch(test_x, test_inst_onehot)[1]
#     print(acc)

# SVM ################################

if True:
    classificator = svm.SVC(gamma='scale')
    classificator.fit(train_x, train_inst)

    pred = classificator.predict(test_x)
    acc = sklearn.metrics.accuracy_score(test_inst, pred)

    print("SVM acc: ", acc)

# kmeans #############################
if False:
    classificator = KMeans(n_clusters=100)
    classificator.fit(train_x, train_inst)

    pred = classificator.predict(test_x)
    acc = sklearn.metrics.accuracy_score(test_inst, pred)

    print("kmeans acc: ", acc)
