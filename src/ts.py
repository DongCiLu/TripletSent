# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the TS dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_mnist.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_CONFIGURATION = "GOOGLE_ADJ_FILTERED2"

if _CONFIGURATION == "FLICKR_128_NORMAL":
    _FILE_PATTERN = 'ts-%s.tfrecord'
    _SPLITS_TO_SIZES = {'train': 285932, 'test': 121738, 'predict': 121738}
    _IMG_SIZE = 128
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 914
    _INPUT_SIZE = 112
elif _CONFIGURATION == "FLICKR_NORMAL":
    _FILE_PATTERN = 'ts-%s_anp.tfrecord' 
    _SPLITS_TO_SIZES = {'train': 282531, 'test': 121288, 'predict': 121288}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 910
    _INPUT_SIZE = 224
elif _CONFIGURATION == "FLICKR_CAP20":
    _FILE_PATTERN = 'ts-%s_anp.tfrecord' 
    _SPLITS_TO_SIZES = {'train': 386619, 'test': 18200, 'predict': 18200}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 910
    _INPUT_SIZE = 224
elif _CONFIGURATION == "FLICKR_NOUN":
    _FILE_PATTERN = 'ts-%s_noun.tfrecord'
    _SPLITS_TO_SIZES = {'train': 283363, 'test': 120813, 'predict': 120813}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 269
    _INPUT_SIZE = 224
elif _CONFIGURATION == "FLICKR_ADJ":
    _FILE_PATTERN = 'ts-%s_adj.tfrecord'
    _SPLITS_TO_SIZES = {'train': 283356, 'test': 120812, 'predict': 120812}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 156
    _INPUT_SIZE = 224
elif _CONFIGURATION == "MNIST":
    _FILE_PATTERN = 'mnist-%s.tfrecord'
    _SPLITS_TO_SIZES = {'train': 60000, 'test': 10000, 'predict': 10000}
    _IMG_SIZE = 28
    _NUM_CHANNELS = 1
    _NUM_CLASSES = 10
    _INPUT_SIZE = 24
elif _CONFIGURATION == "GOOGLE_NORMAL":
    _FILE_PATTERN = 'ts-%s_anp.tfrecord' 
    _SPLITS_TO_SIZES = {'train': 215975, 'test': 93008, 'predict': 93008}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 910
    _INPUT_SIZE = 224
elif _CONFIGURATION == "GOOGLE_NOUN":
    _FILE_PATTERN = 'ts-%s_noun.tfrecord'
    _SPLITS_TO_SIZES = {'train': 215975, 'test': 93008, 'predict': 93008}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 269
    _INPUT_SIZE = 224
elif _CONFIGURATION == "GOOGLE_ADJ":
    _FILE_PATTERN = 'ts-%s_adj.tfrecord'
    _SPLITS_TO_SIZES = {'train': 215975, 'test': 93008, 'predict': 93008}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 156
    _INPUT_SIZE = 224
elif _CONFIGURATION == "GOOGLE_NORMAL_FILTERED1":
    _FILE_PATTERN = 'ts-%s_anp.tfrecord' 
    _SPLITS_TO_SIZES = {'train': 79951, 'test': 34077, 'predict': 34077}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 351
    _INPUT_SIZE = 224
elif _CONFIGURATION == "GOOGLE_NOUN_FILTERED1":
    _FILE_PATTERN = 'ts-%s_noun.tfrecord'
    _SPLITS_TO_SIZES = {'train': 79951, 'test': 34077, 'predict': 34077}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 81
    _INPUT_SIZE = 224
elif _CONFIGURATION == "GOOGLE_ADJ_FILTERED1":
    _FILE_PATTERN = 'ts-%s_adj.tfrecord'
    _SPLITS_TO_SIZES = {'train': 79951, 'test': 34077, 'predict': 34077}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 110
    _INPUT_SIZE = 224
elif _CONFIGURATION == "GOOGLE_NORMAL_FILTERED2":
    _FILE_PATTERN = 'ts-%s_anp.tfrecord' 
    _SPLITS_TO_SIZES = {'train': 46726, 'test': 19765, 'predict': 19765}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 268
    _INPUT_SIZE = 224
elif _CONFIGURATION == "GOOGLE_NOUN_FILTERED2":
    _FILE_PATTERN = 'ts-%s_noun.tfrecord'
    _SPLITS_TO_SIZES = {'train': 46726, 'test': 19765, 'predict': 19765}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 77
    _INPUT_SIZE = 224
elif _CONFIGURATION == "GOOGLE_ADJ_FILTERED2":
    _FILE_PATTERN = 'ts-%s_adj.tfrecord'
    _SPLITS_TO_SIZES = {'train': 46726, 'test': 19765, 'predict': 19765}
    _IMG_SIZE = 256
    _NUM_CHANNELS = 3
    _NUM_CLASSES = 105
    _INPUT_SIZE = 224
else:
    _FILE_PATTERN = '%s'
    _SPLITS_TO_SIZES = {'train': 0, 'test': 0, 'predict': 0}
    _IMG_SIZE = 0
    _NUM_CHANNELS = 0
    _NUM_CLASSES = 0
    _INPUT_SIZE = 0

_ITEMS_TO_DESCRIPTIONS = {
        'image': 'A [_IMG_SIZE x _IMG_SIZE x _NUM_CHANNELS] RGB image.',
        'label': 'A single integer between 0 and _NUM_CLASSES',
        }

def get_split(split_name, dataset_dir, 
        file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading MNIST.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """

    print("****Get split name: {}****".format(split_name))
    sys.stdout.flush()
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    print("Currently using file pattern: {}".format(file_pattern))

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
            'image/encoded': 
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': 
                tf.FixedLenFeature((), tf.string, default_value='raw'),
            'image/class/label': 
                tf.FixedLenFeature([1], tf.int64, 
                default_value=tf.zeros([1], dtype=tf.int64)),
            'image/filename': 
                tf.FixedLenFeature((), tf.string, 
                default_value='no_filename'),
            }

    width = _IMG_SIZE
    height = _IMG_SIZE
    n_channels = _NUM_CHANNELS

    items_to_handlers = {
            'image': slim.tfexample_decoder.Image(
                    shape=[height, width, n_channels], 
                    channels=n_channels),
            'label': slim.tfexample_decoder.Tensor(
                    'image/class/label', shape=[]),
            'filename': slim.tfexample_decoder.Tensor(
                    'image/filename'),
            }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern, 
            reader=reader,
            decoder=decoder,
            num_samples=_SPLITS_TO_SIZES[split_name],
            num_classes=_NUM_CLASSES,
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
            labels_to_names=labels_to_names)
