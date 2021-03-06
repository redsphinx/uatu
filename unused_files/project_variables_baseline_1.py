class ProjectVariable(object):
    def __init__(self):
        """
        Default values for all the experimental variables.
        """
        # which gpu to use
        self._use_gpu = "0" # string
        # the type of cost module. choice of: 'neural_network', 'euclidean', 'euclidean_fc', 'DHSL', 'cosine'
        self._cost_module_type = 'neural_network'  # string
        # the operation to perform with the siamese head features. choice of: 'concatenate', 'add', 'multiply'
        #                                                                     'subtract', 'divide', 'absolute'
        self._neural_distance = 'concatenate'  # string
        # distance threshold
        self._distance_threshold = 0.5
        # make layers of convolutional units 1 and 2 trainable. choice of: True, False
        self._trainable_12 = True  # bool
        # make layers of convolutional units 3 and 4 trainable. choice of: True, False
        self._trainable_34 = True  # bool
        # make layers of convolutional units 5 and 6 trainable. choice of: True, False
        self._trainable_56 = True  # bool
        # make layers of cost module trainable. choice of: True, False
        self._trainable_cost_module = True  # bool
        # make layers of batch normalization layers trainable. choice of: True, False
        self._trainable_bn = True  # bool
        # the number of filters in the convolutional layers. choice of: 1, 2
        self._numfil = 2  # int
        # the type of siamese head to implement. choice of: 'simple', 'batch_normalized'
        self._head_type = 'batch_normalized'# string
        # name of a weights h5 file to initialize the siamese head weights with trained weights from h5 file.
        # note: this is only for the siamese heads
        self._weights_name = None  # string
        # use cyclical learning rate. choice of: True, False
        self._use_cyclical_learning_rate = True  # bool
        # the lower bound for the CLR
        self._cl_min = 0.00001  # float
        # the upper bound for the CLR
        self._cl_max = 0.0001  # float
        # the batch size
        self._batch_size = 64  # int
        # the number of epochs
        self._epochs = 30  # int
        # number of times to repeat experiment
        self._iterations = 5  # int
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
        # the datasets to load. choice of: 'viper', 'cuhk01', 'cuhk02', 'market', 'grid', 'prid450', 'caviar'
        self._datasets = ['viper', 'cuhk02', 'market', 'grid', 'prid450', 'caviar']
        # indicate if we wish to prime the network
        self._priming = False
        # for how many epochs we want to train on the priming train set
        self._prime_epochs = 10
        # name of the model that we want to load
        self._load_model_name = None # string
        # name of the weights that we want to load.
        # note: this is for the entire network
        self._load_weights_name = None # string
        # for priming. if True, indicates to only test. if False, training will happen as well
        self._only_test = False # bool
        # size of kernel in conv2D
        self._kernel = (3, 3) # tuple
        # wether or not to save inbetween
        self._save_inbetween = False # bool
        # at which epoch to save
        self._save_points = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # list
        # what to add to the save file name when saving model and/or weigths
        # choice of 'epoch', 'dataset_name'
        self._name_indication = 'epoch' # string
        # timestamp of when the ranking file was made
        self._ranking_time_name = None # string
        # the ranking number for each dataset
        self._ranking_number = 'half' # str 'half' or actual int
        # train on these datasets. Last dataset gets trained on and tested. the datasets before the last one
        # serve for transfer learning
        self._datsets_order = ['viper', 'cuhk02', 'market', 'grid', 'prid450', 'caviar']
        # UNUSED
        # load all the data for testing
        self._use_all_data = False # bool
        # UNUSED
        # save the weights of the scnn. choice of: True, False
        self._scnn_save_weights_name = None  # string
        # UNUSED
        # save scn as a model. choice of: True, False
        self._scnn_save_model_name = None  # string


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

    # distance threshold
    @property
    def distance_threshold(self):
        return self._distance_threshold

    @distance_threshold.setter
    def distance_threshold(self, value):
        self._distance_threshold = value

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

    # self._weights_name = None  # string
    @property
    def weights_name(self):
        return self._weights_name

    @weights_name.setter
    def weights_name(self, value):
        self._weights_name = value

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

    # self._scnn_save_model_name = None  # string
    @property
    def scnn_save_model_name(self):
        return self._scnn_save_model_name

    @scnn_save_model_name.setter
    def scnn_save_model_name(self, value):
        self._scnn_save_model_name = value

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
        if value == 'all':
            value = ['viper', 'cuhk02', 'market', 'grid', 'prid450', 'caviar']
        self._datasets = value

    # self._priming = False  # bool
    @property
    def priming(self):
        return self._priming

    @priming.setter
    def priming(self, value):
        self._priming = value

    # self._prime_epochs = 10  # int
    @property
    def prime_epochs(self):
        return self._prime_epochs

    @prime_epochs.setter
    def prime_epochs(self, value):
        self._prime_epochs = value

    # self._load_model_name = None  # string
    @property
    def load_model_name(self):
        return self._load_model_name

    @load_model_name.setter
    def load_model_name(self, value):
        self._load_model_name = value

    # self._load_weights_name = None  # string
    @property
    def load_weights_name(self):
        return self._load_weights_name

    @load_weights_name.setter
    def load_weights_name(self, value):
        self._load_weights_name = value

    # for priming. if True, indicates to only test. if False, training will happen as well
    @property
    def only_test(self):
        return self._only_test

    @only_test.setter
    def only_test(self, value):
        self._only_test = value

    # size of kernel in conv2D
    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        self._kernel = value

    # wether or not to save intermediary
    @property
    def save_inbetween(self):
        return self._save_inbetween

    @save_inbetween.setter
    def save_inbetween(self, value):
        self._save_inbetween = value

    # at which epoch to save
    @property
    def save_points(self):
        return self._save_points

    @save_points.setter
    def save_points(self, value):
        self._save_points = value

    # load all the data for testing
    @property
    def use_all_data(self):
        return self._use_all_data

    @use_all_data.setter
    def use_all_data(self, value):
        self._use_all_data = value

    # what to add to the save file name when saving model and/or weigths
    @property
    def name_indication(self):
        return self._name_indication

    @name_indication.setter
    def name_indication(self, value):
        self._name_indication = value

    # timestamp of when the ranking file was made
    @property
    def ranking_time_name(self):
        return self._ranking_time_name

    @ranking_time_name.setter
    def ranking_time_name(self, value):
        self._ranking_time_name = value

    # the ranking number for each dataset
    @property
    def ranking_number(self):
        return self._ranking_number

    @ranking_number.setter
    def ranking_number(self, value):
        self._ranking_number = value

    # train on these datasets. Last dataset gets trained on and tested. the datasets before the last one
    @property
    def datasets_order(self):
        return self._datasets_order

    @datasets_order.setter
    def datasets_order(self, value):
        self._datasets_order = value