#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2017. Vincenzo Lomonaco. All rights reserved.                  #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 31-05-2017                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Single function to extract cnn codes from a pre-trained model given the
    train/test filelists.  """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys, os
import pickle as pkl
sys.path.append("/home/vincenzo/caffe/python")
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
import copy
import sys


def extract_features(
        bpath = '/home/admin/Ior50N/227/',
        train_filelist='/home/admin/data/ior50/ior50n',
        test_filelist='/home/admin/data/ior50/ior50n',
        train_features_file_name='facedb_train_features.save',
        test_features_file_name='facedb_test_features.save',
        net='/home/vincenzo/serverexp/caffe_exp/imagenet/midsize_conv_frozen/'
            'mid_deploy.prototxt',
        weights='/home/vincenzo/data/current_data/imagenet/2012/snapshots/'
                'mid_128+-25%_new_x144_conv_fixed_iter_15000.caffemodel',
        layer_to_extract='fc7_mid',
        mean_path='/home/vincenzo/ssd_data/mean_x128+-50%.npy',
        extract_train=True,
        extract_test=True,
        img_size=128,
        batch_size=60,
        format="pickle",
        verbose=False,
        swap_channels=True,
):
    # Set GPU
    caffe.set_mode_gpu()
    caffe.set_device(0)

    # Make sure that caffe is on the python path:
    caffe_root = '/home/vincenzo/caffe/'

    # Set Caffe to CPU mode, load the net in the test phase for inference,
    # and configure input preprocessing.
    # caffe.set_mode_cpu()
    net = caffe.Net(
        net,
        weights,
        caffe.TEST)

    # Input preprocessing: 'data' is the name of the input
    # blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    # mean pixel
    transformer.set_mean('data', np.load(mean_path).mean(1).mean(1))
    # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_raw_scale('data', 255)

    if swap_channels:
        # the reference model has channels in BGR order instead of RGB
        transformer.set_channel_swap('data', (2,1,0))

    with open(train_filelist, 'r') as ftr:
        train_lines = ftr.readlines()

    with open(test_filelist, 'r') as fte:
        test_lines = fte.readlines()

    num_train_img = len(train_lines)
    num_test_img = len(test_lines)
    num_train_batch = num_train_img // batch_size
    last_batch_size_tr = num_train_img % batch_size
    num_test_batch = num_test_img // batch_size
    last_batch_size_te = num_test_img % batch_size
    print('num_train_img: ', num_train_img)
    print('num_test_img:', num_test_img)
    print('num_train_minibatch: ', num_train_batch)
    print('num_test_minibatch: ', num_test_batch)
    print('last_batch_size_tr: ', last_batch_size_tr)
    print('last_batch_size_te: ', last_batch_size_te)

    # data structures
    training_set = []
    train_labels = []
    train_paths = []
    test_set = []
    test_labels = []
    test_paths = []

    if extract_train:

        # set net to batch size of training images
        net.blobs['data'].reshape(batch_size, 3, img_size, img_size)

        # Feed in the training images.
        print('loading train images...')

        if (num_train_batch == 0):
            current_batch_sz = last_batch_size_tr
        else:
            current_batch_sz = batch_size

        for i, filepath in enumerate(train_lines):
            rel_path, label = filepath.split()
            train_paths.append(rel_path)
            # print i%batch_size
            net.blobs['data'].data[i % batch_size, :, :, :] = \
                transformer.preprocess('data', caffe.io.load_image(bpath + rel_path))
            # print os.path.dirname(filepath).split('/')[-1]
            train_labels.append(int(label))

            if i % (batch_size) == batch_size - 1 and i != 0:
                # Predict saving two layer: layer_to_extract and 'prob'
                if verbose:
                    print('Extracting features, batch ', i // batch_size)
                out = net.forward([layer_to_extract])

                # Loading features as training set
                for j in range(current_batch_sz):
                   #print(out[layer_to_extract][j].shape)
                   training_set.append(copy.copy(out[layer_to_extract][j].flatten()))

                # preparing for the last batch
                # print i, batch_size, ((i+1) / batch_size)
                if ((i + 1) // batch_size) == num_train_batch and last_batch_size_tr != 0:
                    # set net to last batch size of training images
                    net.blobs['data'].reshape(last_batch_size_tr, 3, img_size,
                                              img_size)
                    current_batch_sz = last_batch_size_tr

            elif i == num_train_img - 1:
                #	is the last batch
                print('Extracting features last batch...')
                out = net.forward([layer_to_extract])

                # Loading features as training set
                for j in range(current_batch_sz):
                    training_set.append(copy.copy(out[layer_to_extract][j].flatten()))

    if extract_test:

        # Feed in the test images
        print('loading test images...')

        if (num_test_batch == 0):
            current_batch_sz = last_batch_size_te
        else:
            current_batch_sz = batch_size

        # set net to batch size of test images
        net.blobs['data'].reshape(current_batch_sz, 3, img_size, img_size)

        for i, filepath in enumerate(test_lines):
            rel_path, label = filepath.split()
            test_paths.append(rel_path)
            # print filepath
            net.blobs['data'].data[i % batch_size, :, :,
            :] = transformer.preprocess('data', caffe.io.load_image(bpath + rel_path))
            test_labels.append(int(label))

            if i % batch_size == (batch_size - 1) and i != 0:
                # Predict saving two layer: layer_to_extract and 'prob'
                if verbose:
                    print('Extracting features, batch ', i // batch_size)
                out = net.forward([layer_to_extract])

                # Loading features as training set
                for j in range(current_batch_sz):
                    test_set.append(copy.copy(out[layer_to_extract][j].flatten()))

                # preparing for the last batch
                if (i + 1) // batch_size == num_test_batch and last_batch_size_te != 0:
                    # set net to last batch size of training images
                    net.blobs['data'].reshape(last_batch_size_te, 3, img_size,
                                              img_size)
                    current_batch_sz = last_batch_size_te

            elif i == num_test_img - 1:
                #	is the last batch
                print('Extracting features last batch...')
                out = net.forward([layer_to_extract])

                # Loading features as training set
                for j in range(current_batch_sz):
                    test_set.append(copy.copy(out[layer_to_extract][j].flatten()))

    if extract_train:
        # saving train features and labels
        if format == "pickle":
            f = open(train_features_file_name, 'wb')
            for obj in [training_set, train_labels]:
                pkl.dump(obj, f)
        elif format == "text":
            f = open(train_features_file_name, 'w')
            f.write(str(len(training_set)) + "\n")
            f.write(str(len(training_set[0])) + "\n")
            for path in train_paths:
                path = path.split("/")[-2] + "/" + path.split("/")[-1]
                f.write(path + "\n")
            for row in training_set:
                for item in row:
                    f.write(str(item) + " ")
                f.write("\n")
            for label in train_labels:
                f.write(str(label) + "\n")
        elif format == "fvstxt":
            f = open(test_features_file_name + ".fvstxt", 'w')
            f.write("1\t" + str(len(test_set)) + "\t")
            f.write(str(len(test_set[0])) + "\ntest_set\n")
            for row, label in zip(test_set, test_labels):
                for item in row:
                    f.write(str(item) + "\t")
                f.write(str(label) + "\t\n")
        else:
            print("format unknown.")
            sys.exit(0)

        f.close()

    if extract_test:
        # saving test features and labels
        if format == "pickle":
            f = open(test_features_file_name, 'wb')
            for obj in [test_set, test_labels]:
                pkl.dump(obj, f)
        elif format == "text":
            f = open(test_features_file_name, 'w')
            f.write(str(len(test_set)) + "\n")
            f.write(str(len(test_set[0])) + "\n")
            for path in test_paths:
                path = path.split("/")[-2] + "/" + path.split("/")[-1]
                f.write(path + "\n")
            for row in test_set:
                for item in row:
                    f.write(str(item) + " ")
                f.write("\n")
            for label in test_labels:
                f.write(str(label) + "\n")
        elif format == "fvstxt":
            f = open(test_features_file_name + ".fvstxt", 'w')
            f.write("1\t" + str(len(test_set)) + "\t")
            f.write(str(len(test_set[0])) + "\ntest_set\n")
            for row, label in zip(test_set, test_labels):
                for item in row:
                    f.write(str(item) + "\t")
                f.write(str(label) + "\t\n")
        else:
            print("format unknown.")
            sys.exit(0)

        f.close()

if __name__ == '__main__':

    with open(sys.argv[1], "r") as f:
        cnn = ""
        layer = ""
        hardware = ""

        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            if "cnn" in line:
                cnn = line.split(" ")[1]
            elif "layer_to_extract" in line:
                layer = line.split(" ")[1]
            elif "hardware" in line:
                hardware = line.split(" ")[1]

    bpath = '/home/studente_ml/' + sys.argv[1].split("/")[0] + "/"

    if cnn == "vgg":
        caffemodel = '/home/vincenzo/data/current_data/' \
                     'caffemodels/vgg_face_caffe/VGG_FACE.caffemodel'
        net = '/home/vincenzo/data/current_data/' \
                     'caffemodels/vgg_face_caffe/VGG_FACE_deploy.prototxt'
        img_size = 224
    else:
        caffemodel = '/home/vincenzo/data/current_data/' \
                     'caffemodels/caffeNet/bvlc_reference_caffenet.caffemodel'
        net = '/home/vincenzo/data/current_data/' \
                     'caffemodels/caffeNet/deploy.prototxt'
        img_size = 227

    print(cnn, layer, hardware, net, caffemodel, img_size)

    extract_features(
        bpath = '',
        train_filelist='',
        test_filelist='',
        train_features_file_name=bpath+'train_features.save',
        test_features_file_name=bpath+'test_features.save',
        net=net,
        layer_to_extract=layer,
        weights=caffemodel,
        extract_train=True,
        extract_test=True,
        img_size=img_size,
        format="text"
    )
