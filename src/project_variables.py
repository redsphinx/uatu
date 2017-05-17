class ProjectVariable(object):
    def __init__(self):
        """
        Default values for all the experimental variables.
        """
        # which gpu to use
        self._use_gpu = "0" # string
        # the type of cost module. choice of: 'neural_network', 'euclidean', 'kullback-leibler'
        self._cost_module_type = 'neural_network'  # string
        # the operation to perform with the siamese head features. choice of: 'concatenate', 'add', 'multiply'
        self._neural_distance = 'concatenate'  # string
        # make layers trainable. choice of: True, False
        self._trainable = True  # bool
        # the number of filters in the convolutional layers. choice of: 1, 2
        self._numfil = 2  # int
        # the type of siamese head to implement. choice of: 'simple', 'batch_normalized'
        self._head_type = 'batch_normalized'# string
        # use weights transferred from pre-trained CNN. choice of: True, False
        self._transfer_weights = False  # bool
        # the names of the weights to use for transfer
        self._cnn_weights_name = None  # string
        # use cyclical learning rate. choice of: True, False
        self._use_cyclical_learning_rate = True  # bool
        # the lower bound for the CLR
        self._cl_min = 0.00001  # float
        # the upper bound for the CLR
        self._cl_max = 0.0001  # float
        # the batch size
        self._batch_size = 64  # int
        # the number of epochs
        self._epochs = 1  # int
        # save the weights of the scnn. choice of: True, False
        self._scnn_save_weights_name = None  # string
        # number of times to repeat experiment
        self._iterations = 1  # int
        # the experiment name
        self._experiment_name = None  # string
        # the base learning rate
        self._learning_rate = 0.00001 # float
        # the number of convolutional layers in the siamese head
        self._number_of_conv_layers = None # int
        # the number of units in each dense layer for the 'neural_network' type of cost module
        self._neural_distance_layers = (512, 1024) # tuple
        # (horizontal, vertical) for downscaling with max-pooling. remember shape=(height,width)
        self._pooling_size = [[2, 2], [2, 2]] # list
        # the activation function. choice of: 'relu', 'elu'
        self._activation_function = 'relu' # string
        # the loss function. choice of: 'categorical_crossentropy', 'kullback_leibler_divergence', 'mean_squared_error',
        # 'mean_absolute_error'
        self._loss_function = 'categorical_crossentropy' # string
        # the type of pooling operation. choice of: 'avg_pooling' and 'max_pooling'
        self._pooling_type = 'max_pooling'
        # the datasets to load
        self._datasets = ['viper', 'cuhk01']

    @property
    def use_gpu(self):
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value):
        self._use_gpu = value


    @property
    def cost_module_type(self):
        return self._cost_module_type

    @cost_module_type.setter
    def cost_module_type(self, value):
        self._cost_module_type = value

    # self._neural_distance = None  # string
    @property
    def neural_distance(self):
        return self._neural_distance
    
    @neural_distance.setter
    def neural_distance(self, value):
        self._neural_distance = value
    
    # self._trainable = None  # bool
    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        
    # self._numfil = None  # int
    @property
    def numfil(self):
        return self._numfil

    @numfil.setter
    def numfil(self, value):
        self._numfil = value
        
    # self._head_type = None  # string
    @property
    def head_type(self):
        return self._head_type

    @head_type.setter
    def head_type(self, value):
        self._head_type = value
        
    # self._transfer_weights = None  # bool
    @property
    def transfer_weights(self):
        return self._transfer_weights

    @transfer_weights.setter
    def transfer_weights(self, value):
        self._transfer_weights = value
        
    # self._cnn_weights_name = None  # string
    @property
    def cnn_weights_name(self):
        return self._cnn_weights_name

    @cnn_weights_name.setter
    def cnn_weights_name(self, value):
        self._cnn_weights_name = value
        
    # self._use_cyclical_learning_rate = None  # bool
    @property
    def use_cyclical_learning_rate(self):
        return self._use_cyclical_learning_rate

    @use_cyclical_learning_rate.setter
    def use_cyclical_learning_rate(self, value):
        self._use_cyclical_learning_rate = value
        
    # self._cl_min = None  # float
    @property
    def cl_min(self):
        return self._cl_min

    @cl_min.setter
    def cl_min(self, value):
        self._cl_min = value
        
    # self._cl_max = None  # float
    @property
    def cl_max(self):
        return self._cl_max

    @cl_max.setter
    def cl_max(self, value):
        self._cl_max = value
        
    # self._batch_size = None  # int
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        
    # self._epochs = None  # int
    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        self._epochs = value
        
    # self._scnn_save_weights_name = None  # string
    @property
    def scnn_save_weights_name(self):
        return self._scnn_save_weights_name

    @scnn_save_weights_name.setter
    def scnn_save_weights_name(self, value):
        self._scnn_save_weights_name = value
        
    # self._iterations = None  # int
    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, value):
        self._iterations = value
        
    # self._experiment_name = None  # string
    @property
    def experiment_name(self):
        return self._experiment_name

    @experiment_name.setter
    def experiment_name(self, value):
        self._experiment_name = value

    # self._learning_rate = None  # float
    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    # self._number_of_conv_layers = None  # int
    @property
    def number_of_conv_layers(self):
        return self._number_of_conv_layers

    @number_of_conv_layers.setter
    def number_of_conv_layers(self, value):
        self._number_of_conv_layers = value
        
    # self._neural_distance_layers = None  # array
    @property
    def neural_distance_layers(self):
        return self._neural_distance_layers

    @neural_distance_layers.setter
    def neural_distance_layers(self, value):
        self._neural_distance_layers = value
        
    # self._pooling_size = None  # int
    @property
    def pooling_size(self):
        return self._pooling_size

    @pooling_size.setter
    def pooling_size(self, value):
        self._pooling_size = value
        
    # self._activation_function = None  # string
    @property
    def activation_function(self):
        return self._activation_function

    @activation_function.setter
    def activation_function(self, value):
        self._activation_function = value
        
    # self._loss_function = None  # string
    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    # self._pooling_type = None  # string
    @property
    def pooling_type(self):
        return self._pooling_type

    @pooling_type.setter
    def pooling_type(self, value):
        self._pooling_type = value

    # self._datasets = None  # bool
    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, value):
        self._datasets = value