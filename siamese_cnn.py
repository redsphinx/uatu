import tensorflow as tf
import project_constants as pc
import numpy as np
from PIL import Image


def load_data():
    imarray = np.random.rand(10, 5, 3) * 255
    im1 = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    imarray = np.random.rand(10, 5, 3) * 255
    im2 = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    return (im1, im2)


def build_cnn(data, scope_name):
    # define weights and biases
    conv_1_weights = tf.get_variable(
        'weights_1',
        shape=pc.KERNEL_SHAPE_1,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE
    )

    conv_2_weights = tf.get_variable(
        'weights_2',
        shape=pc.KERNEL_SHAPE_2,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_3_weights = tf.get_variable(
        'weights_3',
        shape=pc.KERNEL_SHAPE_3,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_4_weights = tf.get_variable(
        'weights_4',
        shape=pc.KERNEL_SHAPE_4,
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE,
    )

    conv_5_weights = tf.get_variable(
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
            conv_1_layer = conv_unit(data, conv_1_weights, conv_1_biases)
        with tf.variable_scope('conv_2'):
            conv_2_layer = conv_unit(conv_1_layer, conv_2_weights, conv_2_biases)
        with tf.variable_scope('conv_3'):
            conv_3_layer = conv_unit(conv_2_layer, conv_3_weights, conv_3_biases)
        with tf.variable_scope('conv_4'):
            conv_4_layer = conv_unit(conv_3_layer, conv_4_weights, conv_4_biases)
        with tf.variable_scope('conv_5'):
            conv_5_layer = conv_unit(conv_4_layer, conv_5_weights, conv_5_biases)
    return conv_5_layer



def build_model_siamese_cnn(train_node_1, train_node_2):
    with tf.variable_scope('cnn_models') as scope:
        model_1 = build_cnn(train_node_1, 'cnn_model_1')
        scope.reuse_variables()
        model_2 = build_cnn(train_node_2, 'cnn_model_2')

    # define fc layers
    fc_1_weights = tf.get_variable(
        'fc_1_weights',
        shape=[256, 512],
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE
    )

    fc_2_weights = tf.get_variable(
        'fc_2_weights',
        shape=[512, pc.NUM_CLASSES],
        initializer=tf.contrib.layers.xavier_initializer(),
        dtype=pc.DATA_TYPE
    )
    fc_1_biases = tf.get_variable(
        'fc_1_bias',
        shape=[512],
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    fc_2_biases = tf.get_variable(
        'fc_2_bias',
        shape=[pc.NUM_CLASSES],
        initializer=tf.constant_initializer(0.01),
        dtype=pc.DATA_TYPE
    )

    # get the distance between the 2 features
    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model_1, model_2), 2), 1, keep_dims=True))
    distance = tf.reshape(distance, [1,256])

    fc_1 = tf.nn.relu(tf.matmul(distance, fc_1_weights) + fc_1_biases)
    return tf.matmul(fc_1, fc_2_weights) + fc_2_biases


def main():
    with tf.variable_scope('train-test') as scope:
        data = load_data()
        # training
        train_node_1 = tf.placeholder(pc.DATA_TYPE, shape=[1, 10, 5, 3])
        train_node_2 = tf.placeholder(pc.DATA_TYPE, shape=[1, 10, 5, 3])
        train_label_node = tf.placeholder(pc.DATA_TYPE, shape=[1, 2]) # needs to be OHE

        # validation
        validation_node_1 = tf.placeholder(pc.DATA_TYPE, shape=[1, 10, 5, 3])
        validation_node_2 = tf.placeholder(pc.DATA_TYPE, shape=[1, 10, 5, 3])
        validation_label_node = tf.placeholder(pc.DATA_TYPE, shape=[1])

        logits = build_model_siamese_cnn(train_node_1, train_node_2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits, train_label_node))

        # TODO: maybe put regularizers and dropout

        # step = 0
        step = tf.Variable(0, dtype=pc.DATA_TYPE)
        # Decay once per epoch
        learning_rate = tf.train.exponential_decay(
            pc.START_LEARNING_RATE,
            step * pc.BATCH_SIZE,
            pc.DECAY_STEP,
            pc.DECAY_RATE,
            staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate, pc.MOMENTUM)
        optimizer = optimizer.minimize(loss, global_step=step)

        # Predictions for the current training minibatch.
        train_prediction = tf.nn.softmax(logits)

        # Predictions for the test and validation, which we'll compute less often.
        scope.reuse_variables()
        validation_prediction = tf.nn.softmax(build_model_siamese_cnn(validation_node_1, validation_node_2))

        # running everything
        if tf.__version__ == '0.10.0':
            init = tf.initialize_all_variables()
        elif tf.__version__ == '0.12.0':
            init = tf.global_variables_initializer()
        else:
            print('version not supported. add what to do manually.')
            init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            epoch = 0
            print('yay gabi')


    pass


main()