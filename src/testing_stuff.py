import tensorflow as tf
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from PIL import Image


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
    print keras.__version__


def test_saving():
    test_1 = tf.Variable(tf.constant(10, shape=[10]), name='test_1')
    test_2 = tf.Variable(tf.constant(5, shape=[10]), name='test_2')

    tf.add_to_collection('test', test_1)
    tf.add_to_collection('test', test_2)

    init = tf.global_variables_initializer()
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

        print(sess.run(test_))
        print('\n')
        test_1 = all_vars[0]
        test_2 = all_vars[1]
        print('\n')
        print(sess.run(test_1))
        print(sess.run(test_2))

        try:
            print(sess.run(tf.assert_equal(test_2, all_vars[0])))
        except:
            print('NOT THE SAME')


        print('done')


def test_keras():
    keras.backend.backend()


def test_my_first_keras_model():
    x_train = [1,2,3,4,5,6,7,8,9,0]
    y_train = [0,1,0,1,0,1,0,1,0,1]

    x_test = [3,4,5,6,7]
    y_test = [0,1,0,1,0]

    input_shape = (1,1,10)

    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation='relu'))
    # model.add(Activation('relu'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=1,
              epochs=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=1)
    print('Test score:', score)
    print('Test accuracy:', acc)


def test_making_images():
    thing1 = np.random.rand(200, 100, 3) * 255
    thing2 = np.random.rand(200, 100, 3) * 0
    thing3 = np.random.uniform(low=10, high=10, size=(200,100,3)) *0
    thing4 = np.random.rand(200, 100, 3) * 2
    hor1 = [thing1, thing2]
    hor2 = [thing3, thing4]
    mor = [hor1, hor2]
    img = Image.fromarray(mor[1][0], mode='RGB')
    img.show()




test_making_images()