import tensorflow as tf
import project_constants as pc



def load_data():
    pass


def build_cnn(data, dropout):
    # define weights and biases
    conv_1_weigths = tf.get_variable(
        'weights',
        shape=pc.KERNEL_SHAPE_1,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE
    )

    conv_2_weigths = tf.get_variable(
        'weights',
        shape=pc.KERNEL_SHAPE_2,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_3_weigths = tf.get_variable(
        'weights',
        shape=pc.KERNEL_SHAPE_3,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_4_weigths = tf.get_variable(
        'weights',
        shape=pc.KERNEL_SHAPE_4,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_5_weigths = tf.get_variable(
        'weights',
        shape=pc.KERNEL_SHAPE_5,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_1_biases = tf.get_variable(
        'biases',
        shape=pc.BIAS_SHAPE_1,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_2_biases = tf.get_variable(
        'biases',
        shape=pc.BIAS_SHAPE_2,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_3_biases = tf.get_variable(
        'biases',
        shape=pc.BIAS_SHAPE_3,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_4_biases = tf.get_variable(
        'biases',
        shape=pc.BIAS_SHAPE_4,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    conv_5_biases = tf.get_variable(
        'biases',
        shape=pc.BIAS_SHAPE_5,
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )


    def conv_unit(input, weights, biases):
        conv = tf.nn.conv2d(
            input,
            weights,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        relu = tf.nn.relu(conv, biases)
        pool = tf.nn.max_pool(
            relu,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )
        return pool



    # define model
    def model(data):
        conv_1_layer = conv_unit(data, conv_1_weigths, conv_1_biases)
        conv_2_layer = conv_unit(conv_1_layer, conv_2_weigths, conv_2_biases)
        conv_3_layer = conv_unit(conv_2_layer, conv_3_weigths, conv_3_biases)
        conv_4_layer = conv_unit(conv_3_layer, conv_4_weigths, conv_4_biases)
        conv_5_layer = conv_unit(conv_4_layer, conv_5_weigths, conv_5_biases)

        

    pass


def build_model_siamese_cnn():
    pass


def main():
    pass