#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This script can be used as an interface for the linear svm of scikit
    learn """

import cPickle
import numpy as np
from sklearn import svm

np.set_printoptions(threshold=np.nan)

# TODO: fix copy and paste below
def predict_probs(
        fit_model=True,  # otherwise it will be loaded
        save_model=False,
        train_features_file_name='train_features.save',
        test_features_file_name='test_features.save',
        svm_model_file_name='svm_model.save',
        obj_x_class=1):

    # Loading features
    f = file(train_features_file_name, 'rb')
    training_set = cPickle.load(f)
    train_labels = cPickle.load(f)
    if obj_x_class != 1:
        train_labels = [x / obj_x_class for x in train_labels]
    f.close()

    f = file(test_features_file_name, 'rb')
    test_set = cPickle.load(f)
    test_labels = cPickle.load(f)
    if obj_x_class != 1:
        test_labels = [x / obj_x_class for x in test_labels]
    f.close()

    if fit_model:
        # lin_clf = svm.LinearSVC()
        lin_clf = svm.SVC(probability=True, class_weight='balanced')
        lin_clf.fit(training_set, train_labels)
    else:
        f = file(svm_model_file_name, 'rb')
        lin_clf = cPickle.load(f)
        f.close()

    if save_model:
        # saving model
        f = file(svm_model_file_name, 'wb')
        cPickle.dump(lin_clf, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    # Testing phase
    print "Predicting labels..."
    predicted_labels = lin_clf.predict_proba(test_set)

    return predicted_labels


def compute_accuracy(
        fit_model=True,  # otherwise it will be loaded
        save_model=False,
        train_features_file_name='train_features.save',
        test_features_file_name='test_features.save',
        svm_model_file_name='svm_model.save',
        obj_x_class=1):

    # Loading features
    f = file(train_features_file_name, 'rb')
    training_set = cPickle.load(f)
    train_labels = cPickle.load(f)
    if obj_x_class != 1:
        train_labels = [x / obj_x_class for x in train_labels]
    f.close()

    f = file(test_features_file_name, 'rb')
    test_set = cPickle.load(f)
    test_labels = cPickle.load(f)
    if obj_x_class != 1:
        test_labels = [x / obj_x_class for x in test_labels]
    f.close()

    if fit_model:
        # Learning phase
        # print "Starting training the SVM..."
        # print len(training_set), len(training_set[-1])
        # print len(train_labels)
        # print training_set[-1]
        # print training_set[-1].shape
        # print train_labels
        lin_clf = svm.LinearSVC()
        lin_clf.fit(training_set, train_labels)
    else:
        f = file(svm_model_file_name, 'rb')
        lin_clf = cPickle.load(f)
        f.close()

    if save_model:
        # saving model
        f = file(svm_model_file_name, 'wb')
        cPickle.dump(lin_clf, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    # Testing phase
    print "Testing..."
    # conf_levels = lin_clf.decision_function(test_set)
    # predicted_labels = conf_levels.argmax(1)
    predicted_labels = lin_clf.predict(test_set)
    # print "Predicted_labels: ", predicted_labels
    # print "Test_labels: ", test_labels

    num_right = 0
    # print len(predicted_labels), len(test_labels)
    for i in range(len(test_labels)):
        if predicted_labels[i] == test_labels[i]:
            num_right += 1

    print 'total accuracy: ', num_right / float(len(test_labels))
