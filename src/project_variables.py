class ProjectVariable(object):
    def __init__(self):
        self._cost_module_type = None  # string
        self._neural_distance = None  # string
        self._use_batch_norm = None  # bool
        self._batch_norm_name = None  # string
        self._trainable = None  # bool
        self._numfil = None  # int
        self._head_type = None  # string
        self._transfer_weights = None  # bool
        self._cnn_weights_name = None  # string
        self._use_cyclical_learning_rate = None  # bool
        self._cl_min = None  # float
        self._cm_max = None  # float
        self._batch_size = None  # int
        self._epochs = None  # int
        self._scnn_save_weights_name = None  # string
        self._iterations = None  # int
        self._experiment_name = None  # string


    @property
    def cost_module_type(self):
        return self._cost_module_type

    @cost_module_type.setter
    def cost_module_type(self, value):
        self._cost_module_type = value