import tensorflow as tf
import project_constants as pc
import numpy as np
from PIL import Image


def load_data():
    imarray = np.random.rand(10, 5, 3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    return im


def build_cnn(data, scope_name):
    # define weights and biases
    conv_1_weigths = tf.get_variable(
        'weights_1',
        shape=pc.KERNEL_SHAPE_1,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE
    )

    conv_2_weigths = tf.get_variable(
        'weights_2',
        shape=pc.KERNEL_SHAPE_2,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_3_weigths = tf.get_variable(
        'weights_3',
        shape=pc.KERNEL_SHAPE_3,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_4_weigths = tf.get_variable(
        'weights_4',
        shape=pc.KERNEL_SHAPE_4,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_5_weigths = tf.get_variable(
        'weights_5',
        shape=pc.KERNEL_SHAPE_5,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_1_biases = tf.get_variable(
        'biases_1',
        shape=pc.BIAS_SHAPE_1,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_2_biases = tf.get_variable(
        'biases_2',
        shape=pc.BIAS_SHAPE_2,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_3_biases = tf.get_variable(
        'biases_3',
        shape=pc.BIAS_SHAPE_3,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_4_biases = tf.get_variable(
        'biases_4',
        shape=pc.BIAS_SHAPE_4,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_5_biases = tf.get_variable(
        'biases_5',
        shape=pc.BIAS_SHAPE_5,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    def conv_unit(inp, weights, biases):
        conv = tf.nn.conv2d(
            inp,
            weights,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        relu = tf.nn.relu(tf.nn.bias_add(conv, biases))
        pool = tf.nn.max_pool(
            relu,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )
        return pool

    # define model
    with tf.variable_scope(scope_name):
        with tf.variable_scope('conv_1'):
            conv_1_layer = conv_unit(data, conv_1_weigths, conv_1_biases)
        with tf.variable_scope('conv_2'):
            conv_2_layer = conv_unit(conv_1_layer, conv_2_weigths, conv_2_biases)
        with tf.variable_scope('conv_3'):
            conv_3_layer = conv_unit(conv_2_layer, conv_3_weigths, conv_3_biases)
        with tf.variable_scope('conv_4'):
            conv_4_layer = conv_unit(conv_3_layer, conv_4_weigths, conv_4_biases)
        with tf.variable_scope('conv_5'):
            conv_5_layer = conv_unit(conv_4_layer, conv_5_weigths, conv_5_biases)
    return conv_5_layer



def build_model_siamese_cnn(data):
    train_node = tf.placeholder(pc.DATA_TYPE, shape=[1, 10, 5, 3])

    with tf.variable_scope('cnn_models') as scope:
        model_1 = build_cnn(train_node, 'model_1')
        scope.reuse_variables()
        model_2 = build_cnn(train_node, 'model_2')



    # # define fc layers
    # fc_1_weigths = tf.get_variable(
    #     'fc_weights',
    #     shape=[],
    #     initializer=tf.contrib.layers.xavier_initializer(),
    #     dtype=pc.DATA_TYPE
    # )



    pass


data = np.asarray(load_data(), dtype=np.float32)

print(np.shape(data))
build_model_siamese_cnn(data)


def main():

    pass