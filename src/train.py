# Copyright 2017 The ensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trains a classfier on TS data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import os
import sys
import math
import time
import copy
import pickle
import random
from PIL import Image
from PIL import ImageDraw
# import matplotlib as mpl
# mpl.use("Agg")
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
layers = tf.contrib.layers
from tensorflow.python import debug as tf_debug
from tensorflow.python.framework import ops
import slim
from slim.nets import resnet_v2
from termcolor import colored
from sklearn.neighbors import KNeighborsClassifier

import ts
import data_provider
import triplet_loss

flags = tf.flags

flags.DEFINE_integer('batch_size', 32, 
        'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', '/tmp/mnist/',
        'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_string('data_source', 'tfrecord', 
        'Data source, possible value: tfrecord or nparray.')

flags.DEFINE_string('mode', 'training', 
        'All modes: [training triplet_training visualization].')

flags.DEFINE_string('loss_mode', 'normal', 
        'All loss modes: [normal mix].')

flags.DEFINE_string('network', 'resnet', 
        'Which network to use: alexnet or resnet.')

flags.DEFINE_string('triplet_mining_method', 'batchall', 
        'Which triplet mining method to use: batchall or batchhard.')

flags.DEFINE_float('triplet_margin', 0.2, 
        'Value of margin used for triplet loss.')

flags.DEFINE_float('learning_rate', 1e-3, 
        'Learning rate for the network.')

flags.DEFINE_string('optimizer', 'Adam',
        'Optimizer, possible value: GD, Momentum, Adam.')

flags.DEFINE_integer('num_epochs', 1,
        'The number of training epochs.')

flags.DEFINE_string('data_format', 'NCHW',
        'Data format, possible value: NCHW or NHWC.')

flags.DEFINE_string('visualization_in', 'visualization_input',
        'directories for images to visualize.')

flags.DEFINE_string('visualization_out', 'visualization_results',
        'directories to save visualization images.')

FLAGS = flags.FLAGS

def get_class_list():
    with open(ts._ANP_LIST_FN, 'r') as f:
        class_list = [line.rstrip() for line in f]
    return class_list

def input_fn(split_name, mode='normal', additional_data = None):
    if mode == 'triplet' and split_name == 'train':
        print(colored("Input set as triplet training mode.", 'blue'))
        class_list = get_class_list()
        print(colored("Reading samples from {} classes.".format(
            len(class_list)), 'blue'))
        images, onehot_labels, filenames, axillary_labels = \
                data_provider.provide_triplet_data(split_name, 
                FLAGS.batch_size, FLAGS.dataset_dir, 
                class_list, additional_data)
        # images, onehot_labels, filenames, axillary_labels = \
                # data_provider.provide_data(split_name, 
                # FLAGS.batch_size, FLAGS.dataset_dir, 
                # num_readers=1, num_threads=4)
    elif mode == 'custom_evaluate':
        print(colored("Input set as custom evaluating mode.", 'blue'))
        class_list = get_class_list()
        print(colored("Reading samples from {} classes.".format(
            len(class_list)), 'blue'))
        images, onehot_labels, filenames, axillary_labels = \
                data_provider.provide_triplet_data(split_name, 
                1, FLAGS.dataset_dir, class_list)
    else: #normal
        print(colored("Input set as normal mode.", 'blue'))
        images, onehot_labels, filenames, axillary_labels = \
                data_provider.provide_data(split_name, 
                FLAGS.batch_size, FLAGS.dataset_dir, 
                num_readers=1, num_threads=4)

    # TODO: temporary change
    # Data augmentation.
    if split_name == 'train' and mode != 'custom_evaluate':
        print(colored("Enable data augmentation", 'blue'))
        images = data_provider.data_augmentation(images)
    else: # 'custom evaluate' or 'test'
        print(colored("Disable data augmentation", 'blue'))
        images = tf.image.resize_image_with_crop_or_pad(
                images, ts._INPUT_SIZE, ts._INPUT_SIZE)

    # Change the images to [-1.0, 1.0).
    images = (tf.to_float(images) - 128.0) / 128.0

    print(colored("Tensor formatting check: ", 'blue'))
    print(colored("\timage shape:{}".format(images.shape), 'blue'))
    print(colored("\tonehot label shape:{}".format(
        onehot_labels.shape), 'blue'))
    print(colored("\tfilename shape:{}".format(filenames.shape), 'blue'))
    print(colored("\taxillary label shape:{}".format(
        axillary_labels.shape), 'blue'))
    sys.stdout.flush()

    features = {'images': images, 'filenames': filenames,
                'axillary_labels': axillary_labels}
    return features, onehot_labels 

def alexnet(images, norm_params, mode):
    with tf.contrib.framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=norm_params):
        conv1 = layers.conv2d(images, 96, 11, 4, 
                data_format=FLAGS.data_format)
        print(colored("Conv1 output size: {}".format(conv1.shape), 'blue'))
        pool1 = layers.max_pool2d(conv1, 3, 2,
                # padding='SAME',
                data_format=FLAGS.data_format)
        print(colored("Pooling1 output size: {}".format(pool1.shape), 'blue'))
        conv2 = layers.conv2d(pool1, 256, 5, 1, 
                data_format=FLAGS.data_format)
        print(colored("Conv2 output size: {}".format(conv2.shape), 'blue'))
        pool2 = layers.max_pool2d(conv2, 3, 2,
                # padding='SAME',
                data_format=FLAGS.data_format)
        print(colored("Pooling2 output size: {}".format(pool2.shape), 'blue'))
        conv3 = layers.conv2d(pool2, 384, 3, 1, 
                data_format=FLAGS.data_format)
        print(colored("Conv3 output size: {}".format(conv3.shape), 'blue'))
        conv4 = layers.conv2d(conv3, 384, 3, 1, 
                data_format=FLAGS.data_format)
        print(colored("Conv4 output size: {}".format(conv4.shape), 'blue'))
        conv5 = layers.conv2d(conv4, 256, 3, 1, 
                data_format=FLAGS.data_format)
        print(colored("Conv5 output size: {}".format(conv5.shape), 'blue'))
        pool3 = layers.max_pool2d(conv5, 3, 2,
                # padding='SAME',
                data_format=FLAGS.data_format)
        print(colored("Pooling3 output size: {}".format(pool3.shape), 'blue'))

        flat = layers.flatten(pool3)
        print(colored("Flat layer output size: {}".format(flat.shape), 'blue'))
        fc1 = layers.fully_connected(flat, 4096)
        print(colored("FC1 output size: {}".format(fc1.shape), 'blue'))
        fc1_dropout = layers.dropout(fc1, 
                is_training=(mode==tf.estimator.ModeKeys.TRAIN))
        fc2 = layers.fully_connected(fc1, 4096)
        print(colored("FC2 output size: {}".format(fc2.shape), 'blue'))
        fc2_dropout = layers.dropout(fc2, 
                is_training=(mode==tf.estimator.ModeKeys.TRAIN))
        logits = layers.fully_connected(
                fc2_dropout, ts._NUM_CLASSES, activation_fn=None)
        print(colored("Network output size: {}".format(logits.shape), 'blue'))
        sys.stdout.flush()
        
    return logits

def cnn_model(features, labels, mode):
    images = features['images']
    filenames = features['filenames']
    onehot_labels = labels
    axillary_labels = features['axillary_labels']

    if FLAGS.network == 'alexnet':
        # Format data
        if FLAGS.data_format == 'NCHW':
            print(colored("Converting data format to channels first (NCHW)", \
                    'blue'))
            images = tf.transpose(images, [0, 3, 1, 2])

        # Setup batch normalization
        if mode == tf.estimator.ModeKeys.TRAIN:
            norm_params={'is_training':True, 
                    'data_format': FLAGS.data_format}
        else:
            norm_params={'is_training':False,
                    'data_format': FLAGS.data_format,
                    'updates_collections': None}

        # Create the network
        logits = alexnet(images, norm_params, mode) 

    elif FLAGS.network == 'resnet':
        logits, end_points = resnet_v2.resnet_v2_50(inputs=images, 
                num_classes=ts._NUM_CLASSES, 
                is_training=(mode==tf.estimator.ModeKeys.TRAIN))

    # Inference
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode,
                predictions={
                    'pred_class': predicted_classes,
                    'gt_class': axillary_labels,
                    'embedding': logits,
                    # 'prob': tf.nn.softmax(logits),
                })

    # Training 
    groundtruth_classes = tf.argmax(onehot_labels, 1)
    if FLAGS.mode == "triplet_training":
        if FLAGS.triplet_mining_method == "batchall":
            loss, fraction_positive_triplets, num_valid_triplets = \
                    triplet_loss.batch_all_triplet_loss(
                    axillary_labels, logits, FLAGS.triplet_margin)
        elif FLAGS.triplet_mining_method == "batchhard":
            loss = triplet_loss.batch_hard_triplet_loss(
                    axillary_labels, logits, FLAGS.triplet_margin)
        else:
            "ERROR: Wrong Triplet loss mining method, using softmax"
            loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=onehot_labels, logits=logits)
        if FLAGS.loss_mode == "mix":
            loss += tf.losses.softmax_cross_entropy(
                    onehot_labels=onehot_labels, logits=logits)

    else:
        loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.optimizer == 'GD':
            decay_factor = 0.96
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                    tf.train.get_global_step(),
                    int(math.ceil(float(ts._SPLITS_TO_SIZES['train'] / 
                        FLAGS.batch_size))),
                    decay_factor)
            optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=learning_rate)
        elif FLAGS.optimizer == 'Momentum':
            decay_factor = 0.96
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                    tf.train.get_global_step(),
                    int(math.ceil(float(ts._SPLITS_TO_SIZES['train'] / 
                        FLAGS.batch_size))),
                    decay_factor)
            optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate, momentum=0.9)
        else:
            optimizer = tf.train.AdamOptimizer(
                    learning_rate=FLAGS.learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, 
                    global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)

    # Testing
    # top_5 = tf.metrics.precision_at_top_k(
            # labels=groundtruth_classes, 
            # predictions=predicted_classes,
            # k = 5)
    # top_10 = tf.metrics.precision_at_top_k(
            # labels=groundtruth_classes, 
            # predictions=predicted_classes,
            # k = 10)
    eval_metric_ops = {
            'eval/accuracy': tf.metrics.accuracy(
                labels=groundtruth_classes, 
                predictions=predicted_classes),
            # 'eval/accuracy_top5': top_5,
            # 'eval/accuracy_top10': top_10,
            }
    return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

def load_metadata():
    # noun and adj list are lists of dictionaries for fast look up
    pf = open(ts._METADATA_FN, 'rb')
    noun_list = pickle.load(pf)
    adj_list = pickle.load(pf)
    sample_cnt_list = pickle.load(pf)
    pf.close()
    print(colored("Finished loading noun and adj list of size {} and {}".format(
        len(noun_list), len(adj_list)), 'blue'))
    return noun_list, adj_list, sample_cnt_list

def build_knn_classifier(predictions, noun_list, adj_list): 
    # collect embeddings of all training data
    train_embeddings = []
    train_labels = []
    for pred, _ in zip(predictions, range(ts._SPLITS_TO_SIZES['train'])):
        train_embeddings.append(pred['embedding'])
        train_labels.append(pred['gt_class'])
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_embeddings, train_labels)
    print(colored("Build KNN classifier with {} samples.".format(
        len(train_embeddings)), 'blue'))
    return knn
    
def customized_knn_evaluation(predictions, noun_list, adj_list, knn): 
    accuracy = {'ANP': 0, 'noun': 0, 'adj': 0}
    test_embeddings = []
    test_labels = []
    for pred, _ in zip(predictions, range(ts._SPLITS_TO_SIZES['test'])):
        test_embeddings.append(pred['embedding'])
        test_labels.append(pred['gt_class'])
    test_predictions = knn.predict(test_embeddings)
    for test_prediction, test_label in zip(test_predictions, test_labels):
        correctness = {'ANP': test_prediction == test_label, 
                       'noun': test_prediction in noun_list[test_label],
                       'adj': test_prediction in adj_list[test_label]}
        for key in correctness:
            if correctness[key]:
                accuracy[key] += 1
    for key in accuracy:
        accuracy[key] = float(accuracy[key]) / len(test_predictions)
    print(colored("Final accuracy after {} experiments is:\n{}".format(
        len(test_predictions), accuracy), 'blue'))

def update_tables(noun_dict, sample_cnt_dict, sample):
    if sample not in sample_cnt_dict:
        return
    sample_cnt_dict[sample] -= 2
    if sample_cnt_dict[sample] < 2:
        nbrs = copy.deepcopy(noun_dict[sample].keys())
        if len(nbrs) <= 2:
            for nbr in nbrs:
                sample_cnt_dict.pop(nbr)
                noun_dict.pop(nbr)
        else:
            sample_cnt_dict.pop(sample)
            for nbr in nbrs:
                noun_dict[nbr].pop(sample)
            noun_dict.pop(sample)

def generate_choice_dataset(noun_list, sample_cnt_list):
    choice_dataset = []
    sample_cnt_dict = {}
    noun_dict = {}
    for i in range(len(sample_cnt_list)):
        sample_cnt_dict[i] = copy.deepcopy(sample_cnt_list[i])
    for i in range(len(noun_list)):
        noun_dict[i] = copy.deepcopy(noun_list[i])
    # need a pair of class a and class b so that we can make 4 triplets out of 4 samples
    while sample_cnt_dict:
        # randomly pick anchor and negative
        anchor = random.choice(sample_cnt_dict.keys())
        choice_dataset.append(anchor)
        choice_dataset.append(anchor)
        # negative should not equal to itself
        negative = None
        while negative == None or negative == anchor:
            negative = random.choice(noun_dict[anchor].keys())
        choice_dataset.append(negative)
        choice_dataset.append(negative)
        # update tables
        update_tables(noun_dict, sample_cnt_dict, anchor)
        update_tables(noun_dict, sample_cnt_dict, negative)
    
    print("Genearate choice dataset of {} samples".format(len(choice_dataset)))
    return choice_dataset

def main(_):
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.MakeDirs(FLAGS.train_log_dir)

    tf.set_random_seed(time.time())

    epoch_size = int(math.ceil(float(ts._SPLITS_TO_SIZES['train'] / 
            FLAGS.batch_size)))
    test_size = int(math.ceil(float(ts._SPLITS_TO_SIZES['test'] / 
            FLAGS.batch_size)))

    # TODO: temporary change
    # epoch_size = 5
    # test_size = 10

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    run_config = tf.estimator.RunConfig(
            session_config=session_config,
            save_checkpoints_steps = epoch_size)

    classifier = tf.estimator.Estimator(
            model_fn=cnn_model, 
            model_dir=FLAGS.train_log_dir,
            config=run_config)

    if FLAGS.mode == 'training':
        train_spec = tf.estimator.TrainSpec(input_fn=
                lambda: input_fn('train'),
                max_steps=(FLAGS.num_epochs * epoch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=
                lambda: input_fn('test'),
                steps=test_size,
                throttle_secs=60, start_delay_secs=60)
        tf.estimator.train_and_evaluate(
                classifier, train_spec, eval_spec)

    elif FLAGS.mode == 'triplet_training':
        # 1. load train and test data
        noun_list, adj_list, sample_cnt_list = load_metadata()

        # 2. train the network with triplets
        total_step = 0
        random.seed(time.time())
        for epoch in range(FLAGS.num_epochs):
            # 2.1 prepare dataset sequence for each epoch
            choice_dataset = generate_choice_dataset(noun_list, sample_cnt_list)
            actual_epoch_size = int(math.ceil(float(len(choice_dataset) / 
                FLAGS.batch_size)))
            # 2.2 use the choice dataset to form triplets and train the network
            classifier.train(input_fn=lambda: input_fn(
                    'train', 'triplet', choice_dataset), 
                    # max_steps=(FLAGS.num_epochs * epoch_size))
                    steps=(actual_epoch_size))

        # 3. create a knn classifier with network embeddings of training data
        training_embeddings = classifier.predict(
                input_fn=lambda:input_fn('train', 'custom_evaluate'))
        knn = build_knn_classifier(
                training_embeddings, noun_list, adj_list)

        # 4. get network embeddings of testing data and classify with knn
        testing_embeddings = classifier.predict(
                input_fn=lambda:input_fn('test', 'custom_evaluate'))
        customized_knn_evaluation(
                testing_embeddings, noun_list, adj_list, knn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

