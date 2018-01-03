import os
import tensorflow as tf

def _parse_function(example_proto):
    # features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
    #             "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
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
    
    
def load_flickr_dataset(train_dir, num_epochs, batch_size):
    """Load the flickr dataset.

    :param dataset_dir: path to the dataset directory

    :return: train, test data
    """   
    # remember to make it on CPU
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.map(_decode_function)
    dataset = dataset.map(_resize_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    
    train_filenames = []
    for subdir, dirs, files in os.walk(train_dir):
        for f in files:
            fn = os.path.join(subdir, f)
            print("Collecting data from {}".format(fn))
            train_filenames.append(fn)
    
    with tf.device('/cpu:0'):
        sess = tf.Session()
        sess.run(iterator.initializer, 
                feed_dict={filenames: train_filenames})
            
        print(sess.run(next_element))