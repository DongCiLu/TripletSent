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
import pickle
from PIL import Image
from PIL import ImageDraw
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
layers = tf.contrib.layers
from tensorflow.python import debug as tf_debug
from tensorflow.python.framework import ops

import ts
import data_provider

_NN_BASE_NUM_FILTERS = 1024
_NN_NUM_FC_HIDDEN = 2048
_NUM_CNN_LAYERS = 5

flags = tf.flags

flags.DEFINE_integer('batch_size', 32, 
        'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', '/tmp/mnist/',
        'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_string('mode', 'training', 
        'All modes: [training inference visualization].')

flags.DEFINE_float('lr', 1e-4, 
        'Learning rate for the network.')

flags.DEFINE_string('optimizer', 'Adam',
        'Optimizer, possible value: GD, Momentum, Adam.')

flags.DEFINE_integer('num_epochs', 1,
        'The number of training epochs.')

flags.DEFINE_string('data_format', 'NCHW',
        'Data format, possible value: NCHW or NHWC.')

flags.DEFINE_integer('num_predictions', 1,
        'number of images to predict labels.')

flags.DEFINE_string('prediction_out', 'prediction_results.txt',
        'file to store prediction results.')

flags.DEFINE_string('visualization_in', 'visualization_input',
        'directories for images to visualize.')

flags.DEFINE_string('visualization_out', 'visualization_results',
        'directories to save visualization images.')

FLAGS = flags.FLAGS

def input_fn(split_name):
    if split_name == 'predict':
        batch_size = 1
    else:
        batch_size = FLAGS.batch_size

    images, onehot_labels, filenames, axillary_labels, _ = \
            data_provider.provide_data(split_name, 
            batch_size, FLAGS.dataset_dir)

    print("Tensor formatting check: ")
    print("image shape:{}".format(images.shape))
    print("onehot label shape:{}".format(onehot_labels.shape))
    print("filename shape:{}".format(filenames.shape))
    print("axillary label shape:{}".format(axillary_labels.shape))
    sys.stdout.flush()

    features = {'images': images, 'filenames': filenames,
                'axillary_labels': axillary_labels}
    return features, onehot_labels 

def alex_net(images, norm_params, mode):
    with tf.contrib.framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=norm_params):
        conv1 = layers.conv2d(images, 96, 11, 4, 
                data_format=FLAGS.data_format)
        print("conv layers output size: {}".format(conv1.shape))
        pool1 = layers.max_pool2d(conv1, 3, 2,
                # padding='SAME',
                data_format=FLAGS.data_format)
        print("pooling layers output size: {}".format(pool1.shape))
        conv2 = layers.conv2d(pool1, 256, 7, 1, 
                data_format=FLAGS.data_format)
        print("conv layers output size: {}".format(conv2.shape))
        pool2 = layers.max_pool2d(conv2, 3, 2,
                # padding='SAME',
                data_format=FLAGS.data_format)
        print("pooling layers output size: {}".format(pool2.shape))
        conv3 = layers.conv2d(pool2, 384, 3, 1, 
                data_format=FLAGS.data_format)
        print("conv layers output size: {}".format(conv3.shape))
        conv4 = layers.conv2d(conv3, 384, 3, 1, 
                data_format=FLAGS.data_format)
        print("conv layers output size: {}".format(conv4.shape))
        conv5 = layers.conv2d(conv4, 256, 3, 1, 
                data_format=FLAGS.data_format)
        print("conv layers output size: {}".format(conv5.shape))
        pool3 = layers.max_pool2d(conv5, 3, 2,
                # padding='SAME',
                data_format=FLAGS.data_format)
        print("pooling layers output size: {}".format(pool3.shape))

        flat = layers.flatten(pool3)
        print("after flat layers output size: {}".format(flat.shape))
        fc1 = layers.fully_connected(flat, 4096)
        print("fc layers output size: {}".format(fc1.shape))
        fc1_dropout = layers.dropout(fc1, 
                is_training=(mode==tf.estimator.ModeKeys.TRAIN))
        fc2 = layers.fully_connected(fc1, 4096)
        print("fc layers output size: {}".format(fc2.shape))
        fc2_dropout = layers.dropout(fc2, 
                is_training=(mode==tf.estimator.ModeKeys.TRAIN))
        logits = layers.fully_connected(
                fc2_dropout, ts._NUM_CLASSES, activation_fn=None)
        print("fc layers output size: {}".format(logits.shape))
        sys.stdout.flush()
        
    return logits

def cnn_model(features, labels, mode):
    images = features['images']
    filenames = features['filenames']
    onehot_labels = labels
    axillary_labels = features['axillary_labels']

    # Format data
    if FLAGS.data_format == 'NCHW':
        print("Converting data format to channels first (NCHW)")
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
    logits = alex_net(images, norm_params, mode) 

    # Inference
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode,
                predictions={
                    'pred_class': predicted_classes,
                    'gt_class': axillary_labels,
                    'prob': tf.nn.softmax(logits),
                })

    # Training 
    groundtruth_classes = tf.argmax(onehot_labels, 1)
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.optimizer == 'GD':
            decay_factor = 0.9
            learning_rate = tf.train.exponential_decay(FLAGS.lr,
                    tf.train.get_global_step(),
                    int(math.ceil(float(ts._SPLITS_TO_SIZES['train'] / 
                        FLAGS.batch_size))),
                    decay_factor)
            optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=learning_rate)
        elif FLAGS.optimizer == 'Momentum':
            decay_factor = 0.9
            learning_rate = tf.train.exponential_decay(FLAGS.lr,
                    tf.train.get_global_step(),
                    int(math.ceil(float(ts._SPLITS_TO_SIZES['train'] / 
                        FLAGS.batch_size))),
                    decay_factor)
            optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate, momentum=0.9)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)

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

def load_noun_adj_list():
    pf = open('preprocess/noun_adj_list.p', 'rb')
    noun_list = pickle.load(pf)
    adj_list = pickle.load(pf)
    pf.close()
    return noun_list, adj_list

def customized_evaluation(
        predictions, noun_list, adj_list): 
    accuracy = {'ANP': 0, 'noun': 0, 'adj': 0}
    for pred, _ in zip(predictions, range(FLAGS.num_predictions)):
        pred_ANP = pred['pred_class']
        gt_ANP = pred['gt_class']
        correctness = {'ANP': pred_ANP == gt_ANP, 
                       'noun': pred_ANP in noun_list[gt_ANP],
                       'adj': pred_ANP in adj_list[gt_ANP]}
        for key in correctness:
            if correctness[key]:
                accuracy[key] += 1
    for key in accuracy:
        accuracy[key] = float(accuracy[key]) / FLAGS.num_predictions
    print("Final accuracy after {} experiments is:\n{}".format(
        FLAGS.num_predictions, accuracy))

def main(_):
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.MakeDirs(FLAGS.train_log_dir)

    epoch_size = int(math.ceil(float(ts._SPLITS_TO_SIZES['train'] / 
            FLAGS.batch_size)))
    test_size = int(math.ceil(float(ts._SPLITS_TO_SIZES['test'] / 
            FLAGS.batch_size)))

    '''
    #############Remember to change##############
    epoch_size = 500
    test_size = 10
    #############Remember to change##############
    '''

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

    elif FLAGS.mode == 'custom_training':
        # This is a fake inference model, as we only use the predict
        # function of estimator but not realy do prediction.
        # We use this as a customized evaluation
        noun_list, adj_list = load_noun_adj_list()

        classifier.train(
                input_fn=lambda: input_fn('train'),
                steps=(FLAGS.num_epochs * epoch_size))
        predictions = classifier.predict(
                input_fn=lambda:input_fn('predict'))
        customized_evaluation(
                predictions, noun_list, adj_list)

    elif FLAGS.mode == 'visualization':
        if not tf.gfile.Exists('{}'.format(FLAGS.visualization_out)):
            tf.gfile.MakeDirs('{}'.format(FLAGS.visualization_out))
        visual_input = FLAGS.visualization_in
        for subdir, dirs, files in os.walk(FLAGS.visualization_in):
            for f in files:
                # leave empty
                pass

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()

