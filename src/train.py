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
import math
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
_NUM_CNN_LAYERS = 5

flags = tf.flags

flags.DEFINE_integer('batch_size', 32, 
        'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', '/tmp/mnist/',
        'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_string('mode', 'training', 
        'All modes: [training inference visualization].')

flags.DEFINE_float('lr', 1e-5, 
        'Learning rate for the network.')

flags.DEFINE_integer('max_number_of_steps', 10000,
        'The maximum number of gradient steps.')

flags.DEFINE_integer('max_eval_steps', 20,
        'The maximum number of gradient steps.')

flags.DEFINE_string('data_format', 'NCHW',
        'Data format, possible value: NCHW or NHWC.')

flags.DEFINE_integer('num_predictions', 1,
        'number of images to predict labels.')

flags.DEFINE_string('prediction_out', 'prediction_results',
        'directories to save predicted images.')

flags.DEFINE_string('visualization_in', 'visualization_input',
        'directories for images to visualize.')

flags.DEFINE_string('visualization_out', 'visualization_results',
        'directories to save visualization images.')

FLAGS = flags.FLAGS

def input_fn(split_name):
    images, labels, filenames, _ = data_provider.provide_data(
            split_name, FLAGS.batch_size, FLAGS.dataset_dir)
    features = {'images': images, 'filenames': filenames}
    return (features, labels)

_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)

def cnn_model(features, labels, mode):
    images = features['images']
    filenames = features['filenames']
    # format data
    if FLAGS.data_format == 'NCHW':
        print("Converting data format to channels first (NCHW)")
        images = tf.transpose(images, [0, 3, 1, 2])
    # setup batch normalization
    if mode == tf.estimator.ModeKeys.TRAIN:
        norm_params={'is_training':True, 
                'data_format': FLAGS.data_format}
    else:
        norm_params={'is_training':False,
                'data_format': FLAGS.data_format,
                'updates_collections': None}
    # create the network
    with tf.contrib.framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=_leaky_relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=norm_params):
        # activation_fn=tf.nn.leaky_relu, normalizer_fn=None):
        conv = []
        n_filters = int(_NN_BASE_NUM_FILTERS / (2 ** _NUM_CNN_LAYERS))
        kernel_size = 4
        stride_size = 2
        output_lastlayer = images
        for i in range(_NUM_CNN_LAYERS):
            conv.append(layers.conv2d(output_lastlayer, n_filters, 
                kernel_size, stride_size, 
                data_format=FLAGS.data_format))
            n_filters *= 2
            output_lastlayer = conv[-1]
            print("conv layers output size: {}".format(conv[-1].shape))
        flat = layers.flatten(conv[-1])
        fc1 = layers.fully_connected(flat, _NN_BASE_NUM_FILTERS)
        print("fc layers output size: {}".format(fc1.shape))
        fc1_dropout = layers.dropout(fc1, 
                is_training=(mode==tf.estimator.ModeKeys.TRAIN))
        logits = layers.fully_connected(
                fc1_dropout, ts._NUM_CLASSES, activation_fn=None)
        print("fc layers output size: {}".format(logits.shape))

    # Inference
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode,
                predictions={
                    'class': predicted_classes,
                    'prob': tf.nn.softmax(logits),
                })

    # Training 
    groundtruth_classes = tf.argmax(labels, 1)
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, 
                    global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)

    # Testing
    eval_metric_ops = {
            'eval/accuracy': tf.metrics.accuracy(
                labels=groundtruth_classes, 
                predictions=predicted_classes)
            }
    return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(_):
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.MakeDirs(FLAGS.train_log_dir)

    classifier = tf.estimator.Estimator(
            model_fn=cnn_model, 
            model_dir=FLAGS.train_log_dir)

    if FLAGS.mode == 'training':
        split_name = 'train'
        train_spec = tf.estimator.TrainSpec(input_fn=
                lambda: input_fn(split_name),
                max_steps=FLAGS.max_number_of_steps)
        split_name = 'test'
        eval_spec = tf.estimator.EvalSpec(input_fn=
                lambda: input_fn(split_name),
                throttle_secs=5, start_delay_secs=5)
        tf.estimator.train_and_evaluate(
                classifier, train_spec, eval_spec)

    elif FLAGS.mode == 'inference':
        if not tf.gfile.Exists(FLAGS.prediction_out):
            tf.gfile.MakeDirs(FLAGS.prediction_out)

        split_name = 'predict'
        predictions = classifier.predict(input_fn=
                lambda:input_fn(split_name))
        for pred, _ in zip(predictions, range(FLAGS.num_predictions)):
            print(pred['class'])

    elif FLAGS.mode == 'visualization':
        if not tf.gfile.Exists('{}'.format(FLAGS.visualization_out)):
            tf.gfile.MakeDirs('{}'.format(FLAGS.visualization_out))
        visual_input = FLAGS.visualization_in
        for subdir, dirs, files in os.walk(FLAGS.visualization_in):
            for f in files:
                # leave empty
                pass

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
