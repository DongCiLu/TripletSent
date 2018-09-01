import os
import tensorflow as tf

def _parse_function(example_proto):
    # features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
    #             "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
    }
    parsed_features = tf.parse_single_example(example_proto, feature_map)
    label = tf.cast(parsed_features['image/class/label'], dtype=tf.int32)
    return parsed_features['image/encoded'], label
    
def _decode_function(image_raw, label):
    image_decoded = tf.image.decode_jpeg(image_raw, channels=3)
    return image_decoded, label
                
def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, None])
    image_resized = tf.image.resize_images(image_decoded, [227, 227])
    return image_resized, label

def collect_dataset(data_dir):
    # collecting training filenames
    filenames = []
    for subdir, dirs, files in os.walk(data_dir):
        print("Collecting data from {}".format(subdir))
        for f in files:
            fn = os.path.join(subdir, f)
            filenames.append(fn)
            
    return filenames
    
def input_function(num_epochs, batch_size, tf_graph):
    """dataset graph.

    :param dataset_dir: path to the dataset directory

    :return: iterator initializer hook
    """       
    with tf_graph.as_default():
        # image processing routine
        filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)
        dataset = dataset.map(_decode_function)
        dataset = dataset.map(_resize_function)
        dataset = dataset.shuffle(buffer_size=10000)
        # dataset = dataset.repeat(num_epochs) # move it to train loop
        dataset = dataset.batch(batch_size)
        
        iterator = dataset.make_initializable_iterator()
        # iterator = dataset.make_one_shot_iterator()
        next_image, next_label = iterator.get_next()
        
        # sess = tf.Session()
        # sess.run(iterator.initializer, 
        #         feed_dict={filenames: train_filenames}) 
        # image_1, label_1 = sess.run([next_image, next_label])
       
        return filenames, iterator, next_image, next_label
        