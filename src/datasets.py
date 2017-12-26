import numpy as np
import os

def load_flickr_dataset(dataset_dir):
    """Load the flickr dataset.

    :param dataset_dir: path to the dataset directory

    :return: train, test data
    """
    # Training set
    trX = None
    trY = np.array([])

    for fn in os.listdir(cifar_dir):
        print("loading {}".format(os.path.join(cifar_dir, fn)))
        fo = open(os.path.join(cifar_dir, fn), 'rb')
        data_batch = pickle.load(fo, encoding='latin1')
        fo.close()

        if trX is None:
            trX = data_batch['data']
            trY = data_batch['labels']
        else:
            trX = np.concatenate((trX, data_batch['data']), axis=0)
            trY = np.concatenate((trY, data_batch['labels']), axis=0)

    trX = trX.astype(np.float32) / 255.

    return trX, trY, teX, teY
