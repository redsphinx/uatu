class ProjectVariable(object):
    def __init__(self):
        # the type of cost module. choice of: 'neural_network', 'euclidean'
        self._cost_module_type = None  # string
        # the operation to perform with the siamese head features. choice of: 'concatenate', 'add', 'multiply'
        self._neural_distance = None  # string
        self._trainable = None  # bool
        self._numfil = None  # int
        self._head_type = None  # string
        self._transfer_weights = None  # bool
        self._cnn_weights_name = None  # string
        self._use_cyclical_learning_rate = None  # bool
        self._cl_min = None  # float
        self._cl_max = None  # float
        self._batch_size = None  # int
        self._epochs = None  # int
        self._scnn_save_weights_name = None  # string
        self._iterations = None  # int
        self._experiment_name = None  # string
        self._learning_rate = None # float


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