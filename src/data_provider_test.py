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
import sys
import numpy as np
from PIL import Image
import time

import tensorflow as tf

import data_provider


class DataProviderTest(tf.test.TestCase):
    def test_ts_data_reading(self):
        split_name = 'train'
        batch_size = 8 
        # dataset_dir = "datasets/sentibank_flickr/regular_256/tfrecord"
        dataset_dir = "datasets/google/regular/tfrecord"
        # print (time.time())
        tf.set_random_seed(tf.cast(time.time(), tf.int64))
        '''
        images, oh_labels, filenames, ax_labels, num_samples = \
                data_provider.provide_data(
                split_name, batch_size, dataset_dir, 
                num_readers = 1, num_threads = 1)
        '''
        # whole_dataset_tensors = data_provider.read_whole_dataset(
                 # split_name, dataset_dir)
        images, oh_labels, filenames, ax_labels = \
                data_provider.provide_triplet_data(
                        split_name, batch_size, dataset_dir)

        '''
        with self.test_session() as sess:
            with tf.contrib.slim.queues.QueueRunners(sess):
                whole_dataset_arrays = sess.run(whole_dataset_tensors)
                print("whole dataset reading test: ",  
                        whole_dataset_arrays[0].shape, 
                        whole_dataset_arrays[1].shape, 
                        whole_dataset_arrays[2].shape)
                print(type(whole_dataset_arrays[0]), 
                    whole_dataset_arrays[0].nbytes)
                print(type(whole_dataset_arrays[1]), 
                    whole_dataset_arrays[1].nbytes)
                print(type(whole_dataset_arrays[2]), 
                    whole_dataset_arrays[2].nbytes)
        '''
        with self.test_session() as sess:
            for i in range(2):
                images1, oh_labels1, filenames1, ax_labels1 = \
                        sess.run([images, oh_labels, filenames, ax_labels])

                print(images1)
                print(oh_labels1)
                print(filenames1)
                print(ax_labels1)

if __name__ == '__main__':
    tf.test.main()
