# modified base on https://github.com/blackecho/Deep-Learning-TensorFlow

import os
import errno
import numpy as np

import tensorflow as tf
from tqdm import tqdm

class ConvNet:

    def __init__(self, name, data_shape, layers, n_classes, 
            loss_func='softmax_cross_entropy',
            opt_method='sgd', learning_rate=0.001, dropout=0.5, 
            batch_norm=False, data_format='NCHW', 
            gpu_limit=0.5, tf_graph=tf.Graph()):
        """Constructor.

        :param layers: string used to build the model.
            This string is a comma-separate specification of the layers.
            Supported values:
                conv2d-FX-FY-Z-S: 2d convolution with Z feature maps as output
                    and FX x FY filters. S is the strides size
                maxpool-X: max pooling on the previous layer. X is the size of
                    the max pooling
                full-X: fully connected layer with X units
                softmax: softmax layer
            For example:
                conv2d-5-5-32-1,maxpool-2,conv2d-5-5-64-1,maxpool-2,full-128,full-128,softmax

        :param data_shape: shape of the images in the dataset 
            The parameter is a list in [Height, Width, Channel] (HWC) format
        """
        self.name = name
        self.gpu_limit = gpu_limit
        
        # structure of the network
        self.data_shape = data_shape
        self.n_features = \
                self.data_shape[0] * self.data_shape[1] * self.data_shape[2]
        self.layers = layers
        self.n_classes = n_classes
        
        # hyper-parameters of the network
        self.loss_func = loss_func
        self.opt_method = opt_method
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.data_format = data_format
        
        self.tf_graph = tf_graph
        
        # parameteres of the network
        self.W_vars = None
        self.B_vars = None
        
        # output of the network
        self.mod_y = None
        
        self.cost = None
        self.optimizer = None
        self.train_step = None
        self.accuracy = None
        self.accuracy_5 = None
        self.accuracy_10 = None
        
        self.merged_summary = None
        self.summary_writer = None
        
        # location of logs and models
        self.models_dir = None
        self.logs_dir = None
        self.run_dir = None
        self.model_path = None

    def build_model(self, input_x, input_y):
        with self.tf_graph.as_default():
            # build the graph structure
            self._create_placeholders(input_x, input_y)
            self._create_layers()
            # build the train and evaluation loops
            self._define_cost()
            self._define_train_step()
            self._define_accuracy()
            # setup directories
            self._setup_directories()
        
    def fit(self, num_epochs, batch_size, 
            fn_pholder, iter, trFn, vlFn = None):
        #TODO: add validation interval
        with self.tf_graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_limit)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                self._init_tf_ops(sess)
                
                pbar = tqdm(range(num_epochs))
                for epoch in pbar:
                    sess.run(iter.initializer, 
                            feed_dict={fn_pholder: trFn})
                    run_cnt = 0
                    while True:
                        try:
                            sess.run(self.train_step,
                                    feed_dict = {self.keep_prob: self.dropout,
                                            self.is_training: True})
                            run_cnt += 1
                        except tf.errors.OutOfRangeError:
                            break
                    print("finished the run with {} batchs".format(run_cnt))
                                            
                    if vlFn is not None:
                        validation_predictions = []
                        validation_predictions_5 = []
                        validation_predictions_10 = []
                        sess.run(iter.initializer, 
                                feed_dict={fn_pholder: vlFn})
                        while True:
                            try:
                                result = sess.run([self.merged_summary, self.accuracy, 
                                        self.accuracy_5, self.accuracy_10],
                                        feed_dict = {self.keep_prob: 1,
                                                self.is_training: False})
                                validation_predictions = np.append(
                                        validation_predictions, result[1])
                                validation_predictions_5 = np.append(
                                        validation_predictions, result[2])
                                validation_predictions_10 = np.append(
                                        validation_predictions, result[3])
                            except tf.errors.OutOfRangeError:
                                break
                        accuracy = np.mean(validation_predictions)
                        accuracy_5 = np.mean(validation_predictions_5)
                        accuracy_10 = np.mean(validation_predictions_10)
                        # self.summary_writer.add_summary(result[0], epoch)
                        print("validation accuracy: {}".format(accuracy))
                        print("validation accuracy-5: {}".format(accuracy_5))
                        print("validation accuracy-10: {}".format(accuracy_10))
                        # pbar.set_description("Accuracy: {}".format(accuracy))
                
                tf.train.Saver().save(sess, self.model_path)
                
    def score(self, test_x, test_y):
        with self.tf_graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_limit)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                tf.train.Saver().restore(sess, self.model_path)
                feed_dict = {self.input_data: test_x,
                        self.input_labels: test_y,
                        self.keep_prob: 1,
                        self.is_training: False}
                return self.accuracy.eval(feed_dict)
        
    def _create_placeholders(self, input_x, input_y):
        # self.input_data = tf.placeholder(
                # tf.float32, [None, self.n_features], name='x-input') 
        # self.input_labels = tf.placeholder(
                # tf.float32, [None, self.n_classes], name='y-input')
        self.input_data = input_x
        self.input_labels = tf.one_hot(input_y, self.n_classes)
        self.keep_prob = tf.placeholder(
                tf.float32, name='keep-probs')
        self.is_training = tf.placeholder(
                tf.bool, name='is_training')
            
    def _create_layers(self):
        # assuming the data is arranged in NHWC
        next_layer_feed = tf.reshape(self.input_data,
                [-1, self.data_shape[0], 
                self.data_shape[1],
                self.data_shape[2]])
        prev_output_dim = self.data_shape[2]
        if self.data_format == 'NCHW':
            # Convert from channels_last (NHWC) to channels_first (NCHW). This
            # provides a large training performance boost on NVIDIA GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            print('Converting data shape to channels first (NCHW)')
            next_layer_feed = tf.transpose(next_layer_feed, [0, 3, 1, 2])  
            self.data_shape = [self.data_shape[2],
                    self.data_shape[0],
                    self.data_shape[1]]
            prev_output_dim = self.data_shape[0]

        first_full = True
        
        self.W_vars = []
        self.B_vars = []
        
        for i, l in enumerate(self.layers.split(',')):

            node = l.split('-')
            node_type = node[0]

            if node_type == 'conv2d':

                # ################### #
                # Convolutional Layer #
                # ################### #

                # fx, fy = shape of the convolutional filter
                # feature_maps = number of output dimensions
                fx, fy, feature_maps, stride = int(node[1]),\
                        int(node[2]), int(node[3]), int(node[4])

                print('Building Convolutional layer with '
                        '{} input channels and {} {}x{} filters with stride {}'
                        .format(prev_output_dim, feature_maps, fx, fy, stride))

                # Create weights and biases
                W_conv = tf.Variable(tf.truncated_normal(
                        shape = [fx, fy, prev_output_dim, feature_maps],
                        stddev=0.1))
                B_conv = tf.Variable(tf.constant(0.1, shape = [feature_maps]))
                self.W_vars.append(W_conv)
                self.B_vars.append(B_conv)

                print('shape before conv layer: {}'
                        .format(next_layer_feed.get_shape()))
                strides = [1, stride, stride, 1]
                if self.data_format == 'NCHW':
                    strides = [1, 1, stride, stride]
                # Convolution 
                h_conv = tf.nn.bias_add(tf.nn.conv2d(
                        next_layer_feed, W_conv, 
                        strides = strides,
                        padding = 'SAME',
                        data_format = self.data_format), 
                        B_conv,
                        data_format = self.data_format)
                # h_conv = tf.add(tf.nn.conv2d(
                        # next_layer_feed, W_conv, 
                        # strides = strides,
                        # padding = 'SAME',
                        # data_format = data_format), 
                        # B_conv)
                
                h_batch_norm = h_conv
                if self.batch_norm:
                    h_batch_norm = tf.contrib.layers.batch_norm(h_conv,
                            fused = True,
                            is_training = self.is_training,
                            data_format = self.data_format)
                       
                h_act = tf.nn.relu(h_batch_norm)
                    
                # keep track of the number of output dims of the previous layer
                prev_output_dim = feature_maps
                # output node of the last layer
                next_layer_feed = h_act
                print('output shape from last layer: {}'
                        .format(next_layer_feed.get_shape()))

            elif node_type == 'maxpool':

                # ################# #
                # Max Pooling Layer #
                # ################# #

                ksize_1d, stride = int(node[1]), int(node[2])

                print('Building Max Pooling layer with size {}'.format(ksize_1d))

                ksize = [1, ksize_1d, ksize_1d, 1]
                strides = [1, stride, stride, 1]
                if self.data_format == 'NCHW':
                    ksize = [1, 1, ksize_1d, ksize_1d]
                    strides = [1, 1, stride, stride]                
                    
                next_layer_feed = tf.nn.max_pool(
                        next_layer_feed, 
                        ksize = ksize,
                        strides = strides,
                        padding = 'SAME',
                        data_format = self.data_format)

            elif node_type == 'full':

                # ####################### #
                # Densely Connected Layer #
                # ####################### #
                
                dim = int(node[1])
                fanin = prev_output_dim
                if first_full:  # first fully connected layer
                    shp = next_layer_feed.get_shape()
                    print('***shape before fully connected: {}'.format(shp))
                    tmpx = shp[1].value
                    tmpy = shp[2].value
                    if self.data_format == 'NCHW':
                        tmpx = shp[2].value
                        tmpy = shp[3].value
                    fanin = tmpx * tmpy * prev_output_dim

                print('Building fully connected layer with '
                        '{} in units and {} out units'.format(fanin, dim))
                W_fc = tf.Variable(tf.truncated_normal(
                        shape = [fanin, dim],
                        stddev=0.1))
                B_fc = tf.Variable(tf.constant(0.1, shape = [dim]))
                self.W_vars.append(W_fc)
                self.B_vars.append(B_fc)
                
                h_pool_flat = next_layer_feed
                if first_full:
                    h_pool_flat = tf.reshape(next_layer_feed, [-1, fanin])
                
                h_fc = tf.matmul(h_pool_flat, W_fc) + B_fc
                
                h_batch_norm = h_fc
                if self.batch_norm:
                    h_batch_norm = tf.contrib.layers.batch_norm(h_fc,
                            fused = True,
                            is_training = self.is_training,
                            data_format = self.data_format)
                    
                h_act = tf.nn.relu(h_batch_norm)
                h_act_drop = tf.nn.dropout(h_act, self.keep_prob)

                prev_output_dim = dim
                next_layer_feed = h_act_drop

                first_full = False

                # else:  # not first fully connected layer

                    # dim = int(node[1])
                    # W_fc = self.weight_variable([prev_output_dim, dim])
                    # b_fc = self.bias_variable([dim])
                    # self.W_vars.append(W_fc)
                    # self.B_vars.append(b_fc)

                    # h_fc = tf.nn.relu(tf.add(
                        # tf.matmul(next_layer_feed, W_fc), b_fc))
                    # h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

                    # prev_output_dim = dim
                    # next_layer_feed = h_fc_drop

            elif node_type == 'softmax':

                # ############# #
                # Softmax Layer #
                # ############# #

                print('Building softmax layer with '
                        '{} in units and {} out units'
                        .format(prev_output_dim, self.n_classes))

                W_sm = tf.Variable(tf.truncated_normal(
                        shape = [prev_output_dim, self.n_classes],
                        stddev=0.1))
                b_sm = tf.Variable(tf.constant(0.1, shape = [self.n_classes]))
                self.W_vars.append(W_sm)
                self.B_vars.append(b_sm)

                self.mod_y = tf.matmul(next_layer_feed, W_sm) + b_sm
                
    def _define_cost(self):
        assert self.loss_func in ['softmax_cross_entropy', 'mean_squared_error']
        if self.loss_func == 'softmax_cross_entropy':
            self.cost = tf.losses.softmax_cross_entropy(self.input_labels, self.mod_y)
        elif self.loss_func == 'mean_squared_error':
            self.cost = tf.losses.mean_squared_error(self.input_labels, self.mod_y)
        tf.summary.scalar(self.loss_func, self.cost)
        
    def _define_train_step(self):
        assert self.opt_method in ['sgd', 'adam', 'momentum']
        if self.opt_method == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.opt_method == "adam":
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
        elif self.opt_method == "momentum":
            momentum = 0.9
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum)
        
        # updates for the batch normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = self.optimizer.minimize(self.cost)
        
    def _define_accuracy(self):
        mod_pred = tf.argmax(self.mod_y, 1)
        correct_pred = tf.equal(mod_pred, tf.argmax(self.input_labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        correct_pred_5 = tf.nn.in_top_k(self.mod_y, tf.argmax(self.input_labels, 1), 5)
        self.accuracy_5 = tf.reduce_mean(tf.cast(correct_pred_5, tf.float32))
        tf.summary.scalar('accuracy-5', self.accuracy_5)
        correct_pred_10 = tf.nn.in_top_k(self.mod_y, tf.argmax(self.input_labels, 1), 10)
        self.accuracy_10 = tf.reduce_mean(tf.cast(correct_pred_10, tf.float32))
        tf.summary.scalar('accuracy-10', self.accuracy_10)
        
    def _setup_directories(self):
        home_dir = os.path.join(os.path.expanduser("~"), '.zluconvnet')
        self.models_dir = os.path.join(home_dir, 'models/')
        self.logs_dir = os.path.join(home_dir, 'logs/')
        self.mkdir_p(self.models_dir)
        self.mkdir_p(self.logs_dir)  
        self.model_path = os.path.join(self.models_dir, self.name)
    
    def _init_tf_ops(self, sess):
        self.merged_summary = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        
        sess.run(init_op)
        
        run_id = 0
        for e in os.listdir(self.logs_dir):
            if e[:3] == 'run':
                r = int(e[3:])
                if r > run_id:
                    run_id = r
        run_id += 1
        self.run_dir = os.path.join(self.logs_dir, 'run' + str(run_id))
        print('Tensorboard logs dir for this run is {}'.format(self.run_dir))
        self.summary_writer = tf.summary.FileWriter(self.run_dir, sess.graph)
        
    @staticmethod
    def mkdir_p(path):
        """Recursively create directories."""
        try:
            os.makedirs(path)
        except OSError as exc:  # Python > 2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise
            
    @staticmethod
    def get_batches(data, batch_size):
        for i in range(0, data.shape[0], batch_size):
            yield data[i : i + batch_size]
