# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for data_provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
import time

import tensorflow as tf

import data_provider


class DataProviderTest(tf.test.TestCase):
    def test_celegans_data_reading(self):
        split_name = 'train'
        batch_size = 10 
        dataset_dir = "datasets/sentibank_flickr/regular_256/tfrecord"
        print (time.time())
        tf.set_random_seed(time.time())
        images, oh_labels, filenames, \
                ax_labels, num_samples, boxes, rotate_degree = \
                data_provider.provide_data(
                split_name, batch_size, dataset_dir, 
                num_readers = 1, num_threads = 1)

        with self.test_session() as sess:
            with tf.contrib.slim.queues.QueueRunners(sess):
                images, oh_labels, filenames, \
                        ax_labels, boxes, rotate_degree = \
                        sess.run([images, oh_labels, filenames, 
                        ax_labels, boxes, rotate_degree])
                print(rotate_degree)

if __name__ == '__main__':
    tf.test.main()
