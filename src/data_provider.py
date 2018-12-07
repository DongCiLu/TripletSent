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
"""Contains code for loading and preprocessing the MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import math
import tensorflow as tf

# from slim.datasets import dataset_factory as datasets
import ts

_DA_ROTATE_LIMIT = 30
_LEN_LIMIT = 1 - 2.0 * (ts._IMG_SIZE - ts._INPUT_SIZE) / ts._IMG_SIZE

slim = tf.contrib.slim

def data_augmentation(image):
    batch_size = image.shape[0]
    # flip image
    image = tf.image.random_flip_left_right(image)
    # rotate small angles
    # rotate_degree = tf.random_uniform(
             # [], -_DA_ROTATE_LIMIT, _DA_ROTATE_LIMIT)
    # image = tf.contrib.image.rotate(
            # image, rotate_degree * math.pi / 180, 'BILINEAR')
    '''
    # zoom in or out images
    l = tf.random_uniform([], _LEN_LIMIT, 1)
    x = tf.random_uniform([], 0, 1 - l)
    y = tf.random_uniform([], 0, 1 - l)
    boxes = tf.stack([y, x, y + l, x + l])
    crop_size = [ts._INPUT_SIZE, ts._INPUT_SIZE]
    image = tf.image.crop_and_resize(
            tf.expand_dims(image, 0), 
            tf.expand_dims(boxes, 0), 
            [0], crop_size)
    image = tf.squeeze(image)
    '''
    # random crop images
    image = tf.random_crop(image, 
            # [ts._INPUT_SIZE, ts._INPUT_SIZE, ts._NUM_CHANNELS])
            [batch_size, ts._INPUT_SIZE, ts._INPUT_SIZE, ts._NUM_CHANNELS])

    return image

def provide_triplet_data(split_name, batch_size, dataset_dir):
    ''' 
    Arrange data into batches that each batch contains multiple class examples

    Args:
      source: data structure that holds all the training data

    Returns:
      images: A `Tensor` of size [batch_size, 256, 256, 1]
      one_hot_labels: A `Tensor` of size [batch_size, mnist.NUM_CLASSES], where
        each row has a single element set to one and the rest set to zeros.
      num_samples: The number of total samples in the dataset.
    '''

    file_pattern = os.path.join(dataset_dir, ts._FILE_PATTERN % split_name)
    dataset = tf.data.TFRecordDataset(file_pattern)
    dataset = dataset.map(ts.parse_example)
    dataset_size = ts._SPLITS_TO_SIZES[split_name]
    dataset = dataset.shuffle(dataset_size)
    # dataset = dataset.shuffle(1000)
    dataset1 = dataset.filter(lambda image, label, filename: 
            tf.reshape(tf.equal(tf.unstack(label)[0], 177), []))
    dataset2 = dataset.filter(lambda image, label, filename: 
            tf.reshape(tf.equal(tf.unstack(label)[0], 99), []))

    datasets = [dataset1, dataset2]

    '''
    dataset = tf.data.Dataset.zip((dataset1, dataset2)).flat_map(
            lambda x0, x1: tf.data.Dataset.from_tensors(x0).concatenate(
                tf.data.Dataset.from_tensors(x1)))
    # dataset = tf.data.Dataset.range(2).interleave(
            # lambda x: datasets[x], 
            # cycle_length=2, block_length=1) 
            # num_parallel_calls=None)
    '''
    choice_dataset = tf.data.Dataset.range(2).repeat()
    dataset = tf.data.experimental.choose_from_datasets(
            datasets, choice_dataset)

    dataset = dataset.batch(batch_size, True) # discard tail
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    print("**********************{}".format(next_element))

    images, labels, filenames = \
            next_element[0], next_element[1], next_element[2]
    labels = tf.squeeze(labels)

    one_hot_labels = tf.one_hot(labels, ts._NUM_CLASSES)

    return images, one_hot_labels, filenames, labels

def provide_data(split_name, batch_size, 
        dataset_dir, num_readers=1, num_threads=1):
    """Provides batches of MNIST digits.

    Args:
      split_name: Either 'train' or 'test'.
      batch_size: The number of images in each batch.
      dataset_dir: The directory where the MNIST data can be found.
      num_readers: Number of dataset readers.
      num_threads: Number of prefetching threads.

    Returns:
      images: A `Tensor` of size [batch_size, 256, 256, 1]
      one_hot_labels: A `Tensor` of size [batch_size, mnist.NUM_CLASSES], where
        each row has a single element set to one and the rest set to zeros.
      num_samples: The number of total samples in the dataset.

    Raises:
      ValueError: If `split_name` is not either 'train' or 'test'.
    """
    dataset = ts.get_split(split_name, dataset_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_readers,
            common_queue_capacity=2 * batch_size,
            common_queue_min=batch_size,
            shuffle=(split_name == 'train'))
    [image, label, filename] = \
            provider.get(['image', 'label', 'filename'])

    '''
    # Data augmentation.
    if split_name == 'train':
        print("enable data augmentation")
        image = data_augmentation(image)
    else: # 'predict' or 'test'
        print("central crop testing data")
        image = tf.image.resize_image_with_crop_or_pad(
                image, ts._INPUT_SIZE, ts._INPUT_SIZE)

    # Change the images to [-1.0, 1.0).
    image = (tf.to_float(image) - 128.0) / 128.0
    '''

    # Creates a QueueRunner for the pre-fetching operation.
    images, labels, filenames = tf.train.batch(
            [image, label, filename],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=5 * batch_size)

    one_hot_labels = tf.one_hot(labels, dataset.num_classes)
    return images, one_hot_labels, filenames, labels

def float_image_to_uint8(image):
    """Convert float image in [-1, 1) to [0, 255] uint8.

    Note that `1` gets mapped to `0`, but `1 - epsilon` gets mapped to 255.

    Args:
      image: An image tensor. Values should be in [-1, 1).

    Returns:
      Input image cast to uint8 and with integer values in [0, 255].
    """
    image = (image * 128.0) + 128.0
    return tf.cast(image, tf.uint8)
