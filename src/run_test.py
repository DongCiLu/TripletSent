import argparse
import numpy as np
import tensorflow as tf

import conv_net
import datasets

def to_one_hot(dataY):
    """Convert the vector of labels dataY into one-hot encoding.

    :param dataY: vector of labels
    :return: one-hot encoded labels
    """
    nc = 1 + np.max(dataY)
    onehot = [np.zeros(nc, dtype=np.int8) for _ in dataY]
    for i, j in enumerate(dataY):
        onehot[i][j] = 1
    return onehot

def run_cnn(dataset_dir, para_num_epochs):   
    # common parameters
    gpu_limit = 0.4
    data_dir = dataset_dir
    num_epochs = para_num_epochs
    batch_size = 64
    
    # parameters for cnn
    name = 'test_cnn'
    data_shape = '227,227,3'
    # data_shape = '32,32,3'
    layers = 'conv2d-11-11-96-4,maxpool-3-2,conv2d-5-5-256-1,maxpool-3-2,conv2d-3-3-384-1,conv2d-3-3-384-1,conv2d-3-3-256-1,maxpool-3-2,full-4096,full-4096,softmax'
    # layers = 'conv2d-5-5-16-1,maxpool-2-2,conv2d-5-5-64-1,maxpool-2-2,full-1024,softmax'
    n_classes = 1553
    # n_classes = 10
    loss_func = 'softmax_cross_entropy'
    # opt_method = 'adam'
    opt_method = 'sgd'
    learning_rate = 1e-4
    dropout = 0.5
    batch_norm = True
    # batch_norm = False
    data_format = 'NCHW'
    
    tf_graph = tf.Graph()
    
    # prepare data
    trFn = datasets.collect_dataset(data_dir + "/train")
    vlFn = datasets.collect_dataset(data_dir + "/val")
    fn_pholder, iter, image, label = datasets.input_function(
            num_epochs, batch_size, tf_graph)
    
    # define Convolutional Network
    cnn = conv_net.ConvNet(name=name,
            data_shape=[int(i) for i in data_shape.split(',')],
            layers=layers, n_classes=n_classes, 
            loss_func=loss_func, opt_method=opt_method,
            learning_rate=learning_rate, dropout=dropout, 
            batch_norm=batch_norm, data_format=data_format, 
            gpu_limit=gpu_limit, tf_graph=tf_graph)
    
    print('Build Convolutional Network...')
    cnn.build_model(image, label)
    
    print('Start Convolutional Network training...')
    # cnn.fit(num_epochs, batch_size, trX, trY, vlX, vlY)
    cnn.fit(num_epochs, batch_size, fn_pholder, iter, trFn, vlFn)
    
    print('Run test set on the trained model...')
    # print(cnn.score(teX, teY))
    
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--num_epochs', type=int, default=3, required=False)
    args = arg_parser.parse_args()
    dataset_dir = 'datasets/sentibank_flickr/preprocessed'
    
    run_cnn(dataset_dir, args.num_epochs)
