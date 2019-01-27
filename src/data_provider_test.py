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
import pickle
import random
from PIL import Image
import time

import tensorflow as tf

import data_provider
import ts
import train

class DataProviderTest(tf.test.TestCase):
    def test_ts_data_reading(self):
        split_name = 'train'
        batch_size = 16 
        dataset_dir = "datasets/google/regular/tfrecord"
        tf.set_random_seed(tf.cast(time.time(), tf.int64))
        random.seed(time.time())

        noun_list, adj_list, sample_cnt_list = train.load_metadata()
        choice_dataset = train.generate_choice_dataset(
                noun_list, sample_cnt_list)
        class_list = train.get_class_list()
        images, oh_labels, filenames, ax_labels = \
                data_provider.provide_triplet_data(
                        split_name, batch_size, dataset_dir, 
                        class_list, choice_dataset)

        with self.test_session() as sess:
            for i in range(2):
                images1, oh_labels1, filenames1, ax_labels1 = \
                        sess.run([images, oh_labels, filenames, ax_labels])

                # print(images1)
                print(oh_labels1)
                print(filenames1)
                print(ax_labels1)

if __name__ == '__main__':
    tf.test.main()
