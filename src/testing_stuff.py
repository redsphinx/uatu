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


def test_saving():
    test_1 = tf.Variable(tf.constant(10, shape=[10]), name='test_1')
    test_2 = tf.Variable(tf.constant(5, shape=[10]), name='test_2')
    tf.add_to_collection('test', test_1)
    tf.add_to_collection('test', test_2)

    init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(test_1))
        print(sess.run(test_2))
        saver.save(sess, 'my_test')


def test_restoring():
    test_ = tf.Variable(tf.constant(10, shape=[10]), name='test_')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        new_saver = tf.train.import_meta_graph('my_test.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        all_vars = tf.get_collection('test')
        # for t in all_vars:
        #     thing = sess.run(t)
        #     print(thing)
        print(sess.run(test_))
        print('\n')
        test_1 = all_vars[0]
        test_2 = all_vars[1]
        print('\n')
        print(sess.run(test_1))
        print(sess.run(test_2))

        # print(sess.run(tf.assert_equal(test_1, test_)))
        # print(sess.run(tf.assert_equal(test_2, all_vars[3])))
        try:
            print(sess.run(tf.assert_equal(test_2, all_vars[0])))
        except:
            print('NOT THE SAME')


        print('done')


test_saving()
test_restoring()