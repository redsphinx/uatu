import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def test_data_pipeline():
    path = '/home/gabi/Documents/datasets/humans/1/per00001.jpg'

    file_queue = tf.train.string_input_producer([path])
    print(type(file_queue))

    reader = tf.WholeFileReader()
    print(type(reader))

    key, value = reader.read(file_queue)
    print(key)
    print(value)

    my_img = tf.image.decode_png(value)
    print(type(my_img))


def test_perm():
    x_before = tf.constant([[[1, 2, 3], [4,5,6]], [[11, 22, 33], [44,55,66]]])
    x_after_1 = tf.transpose(x_before, perm=[1, 0, 2])
    x_after_2 = tf.transpose(x_before, perm=[0, 2, 1])
    x_after_3 = tf.transpose(x_before, perm=[0, 1, 2])
    x_after_4 = tf.transpose(x_before, perm=[1,2,0])

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(str('original: ')+ '\n' + str(tf.Tensor.eval(x_before)) + '\n')
        print(str('1 0 2 ') + '\n'+ str(tf.Tensor.eval(x_after_1))+ '\n')
        print(str('0 2 1 ') + '\n'+ str(tf.Tensor.eval(x_after_2))+ '\n')
        print(str('0 1 2 ') + '\n'+ str(tf.Tensor.eval(x_after_3))+ '\n')
        print(str('1 2 0 ') + '\n'+ str(tf.Tensor.eval(x_after_4))+ '\n')


def test_reshape():
    r = tf.constant([[1, 2, 3], [4,5,6]])
    r_after = tf.reshape(r, [-1])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(tf.Tensor.eval(r_after))


def tupconv(lst):
    tuplst = []
    for x in lst:
        if isinstance(x, list):
            tuplst.append(tupconv(x))
        else:
            tuplst.append(x)


def get_version():
    ver = tf.__version__
    print(type(ver))
    print(tf.__version__)


test_perm()

