import tensorflow as tf
import os
# import keras
# from keras.models import Sequential, Model
# from keras.layers import Dense, Dropout, Activation, LSTM, Embedding, Input
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from PIL import Image
# from keras.models import load_model
# import project_constants as pc
import os
from tensorflow.python.client import device_lib
import h5py

# from imblearn.datasets import make_imbalance
import sys
import time
from scipy import ndimage
from scipy.io import loadmat
import matplotlib.pyplot as plt
from skimage.util import random_noise
from matplotlib.image import imsave
from numpy.linalg import inv
import urllib
import zipfile


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
    # print keras.__version__


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


def test_multiple_inputs():
    # Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
    # Note that we can name any layer by passing it a "name" argument.
    main_input = Input(shape=(100,), dtype='int32', name='main_input')

    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out = LSTM(32)(x)

    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

    auxiliary_input = Input(shape=(5,), name='aux_input')
    x = keras.layers.concatenate([lstm_out, auxiliary_input])

    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])


def test_load_model():
    model_location = os.path.join(pc.SAVE_LOCATION_MODEL_WEIGHTS, 'cnn_model.h5')
    model = load_model(model_location)
    print('Model loaded.')

    model.summary()


# @profile
def test_arr_vs_list():
    arr = np.zeros((9,9,9,9))
    a = range(9)
    b = a
    for i in range(8):
        b = [b, a]
    c = b
    for i in range(8):
        c = [c, b]

    lsit = [c,c,c,c,c,c,c,c,c]
    thingy1 = arr[0,0,0,0]
    arr = np.append(arr, 1)

    asd = np.array([])
    asd = np.append(asd, 1)

    qwe = []
    qwe.append(1)
    thingy2 = lsit[0][0][0][0]

    lsit.append(1)

    del arr
    del lsit


def test():
    c = 0
    for a in range(100):
        c += 1
    return c


def test_2(a=True):
    d = 4 if a else 5
    return d


def thing():
    a=test()
    c=test_2()
    return a*c


def test_path():
    path = pc.LOCATION_RAW_CUHK01
    the_list = os.listdir(path)
    print(len(the_list))


def assign_experiments():
    import nvidia_bash_utils as bu
    import running_experiments as re
    # list_of_experiments = []
    list_of_experiments = ['experishit']
    # for i in range(57, 69 + 1):
    #     list_of_experiments.append('experiment_%d' % i)
    gpu_number = 0
    # +1 because there's a difference in CUDA_VISIBLE_DEVICES and the numbering of GPUs in nvidia-smi
    if bu.is_gpu_busy(gpu_number + 1) == False:
        print(list_of_experiments)
        the_experiment = getattr(re, list_of_experiments.pop(0))
        print(list_of_experiments)
        the_experiment(gpu_number)


def dict_to_h5():
    image_path = '/home/gabi/Documents/datasets/CUHK/cropped_CUHK1/images/'
    img_dict = {}

    all_images = os.listdir(image_path)[0:10]

    for i in range(10):
        the_key = 'img+_%d' % i
        the_img = ndimage.imread(os.path.join(image_path, all_images[i]))
        img_dict[the_key] = the_img

    h5_path = 'test_dict_with_slash.h5'
    with h5py.File(h5_path, 'w') as myfile:
        for i in range(10):
            the_key = img_dict.keys()[i]
            data = myfile.create_dataset(name=the_key, data=img_dict[the_key])

    print('asdf')


def read_h5dict():
    hdf5_file = h5py.File('test_dict.h5', 'r')
    for i in range(10):
        thing = hdf5_file['img_1'][:]
        print('asdf')


def get_names():
    hdf5_file_3 = h5py.File('/home/gabi/PycharmProjects/uatu/src/test_dict_with_slash.h5', 'r')
    hdf5_file_2 = h5py.File('/home/gabi/PycharmProjects/uatu/src/test_dict.h5', 'r')
    hdf5_file = h5py.File('/home/gabi/PycharmProjects/uatu/data/GRID/grid.h5', 'r')

    print(hdf5_file.name)
    b = hdf5_file_2.keys()
    a = hdf5_file.keys()
    c = hdf5_file_3.keys()
    print(a)

    # / home / gabi / PycharmProjects / uatu / data / VIPER / viper.h5
    for i in range(10):
        thing = hdf5_file['img_%i' % i][:]
        other = hdf5_file['fake'][:]
        print('asdf')
    print('asdf')


def noise():
    the_image = ndimage.imread('/home/gabi/Documents/datasets/INRIAPerson/train_64x128_H96/real_cropped_images_pos/crop001001a.png')

    the_image_2 = the_image[:, :, 0:3]

    image_2 = random_noise(the_image)

    image_2_2 = random_noise(the_image_2)

    # plt.imshow(image_2)
    # plt.imshow(image_2_2)

    imsave('crap.png', image_2_2)

    return image_2


def load_image():
    the_image = ndimage.imread('crap.png')

    new_image = the_image[:, :, 0:3]

    plt.imshow(new_image)
    print('asdf')

def test_file_name():
    file_name = os.path.basename(__file__)
    print(file_name)


def kissme():
    a = np.array([5, 5, 5])
    b = np.array([3, 0, 6])
    c = np.array([2, 3, 4])
    d = np.array([1, 2, 3])

    e = np.array([4, 4, 4])
    f = np.array([4, 5, 6])

    # pairs = [(a, b),(c, d),(b, c), (a, e), (d, f)]
    # labels = [0, 1, 0, 1, 1]

    pairs = [(a, b), (c, d), (b, c)]
    labels = [0, 1, 0]

    match = np.zeros((len(a), len(a)))
    mismatch = np.zeros((len(a), len(a)))

    pairs_match = []
    pairs_mismatch = []
    for item in range(len(labels)):
        if labels[item]:
            pairs_match.append(pairs[item])
        else:
            pairs_mismatch.append(pairs[item])

    for item in range(len(pairs_match)):
        diff = pairs_match[item][0] - pairs_match[item][1]
        mult = np.outer(diff, diff)
        match += mult

    for item in range(len(pairs_mismatch)):
        diff = pairs_mismatch[item][0] - pairs_mismatch[item][1]
        mult = np.outer(diff, diff)
        mismatch += mult

    # match = inv(match)
    # mismatch = inv(mismatch)

    match /= len(pairs_match)
    mismatch /= len(pairs_mismatch)

    m = match - mismatch

    d_a_b = np.inner(np.inner((a - b), m), (a - b))
    d_c_d = np.inner(np.inner((c - d), m), (c - d))
    d_b_c = np.inner(np.inner((b - c), m), (b - c))

    d_a_e = np.inner(np.inner((a - e), m), (a - e))
    d_d_f = np.inner(np.inner((d - f), m), (d - f))

    # TODO do the clipping of spectrum thing


    print(d_a_b)
    print(d_c_d)
    print(d_b_c)
    print(d_a_e)
    print(d_d_f)


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.iteritems():
        print "    %s: %s" % (key, val)

def test_load_mat():
    f = h5py.File('/home/gabi/Documents/datasets/CUHK/CUHK3/cuhk-03.mat', 'r')
    # detected
    # MATLAB_class: cell
    # labeled
    # MATLAB_class: cell
    # testsets
    # MATLAB_class: cell
    data = f.get('detected')


    data.visititems(print_attrs)

    # data = np.array(data)

    # thing = loadmat('/home/gabi/Documents/datasets/CUHK/CUHK3/cuhk-03.mat')
    print('asdf')

def get_duplicate():
    path = '/home/gabi/PycharmProjects/uatu/data/CUHK02/P3/id_all_file.txt'
    unique = '/home/gabi/PycharmProjects/uatu/data/CUHK02/P3/unique_id_file.txt'

    a_list = list(np.genfromtxt(path, dtype=None))
    u_list = list(np.genfromtxt(unique, dtype=None))
    print(len(a_list))

    tally = 0
    for ind in range(len(a_list)):
        if a_list[ind] in u_list:
            tally +=1
        if tally == 3 and (not a_list[ind+1] == a_list[ind]):
            print(a_list[ind])
        if tally == 4 and (not a_list[ind+1] == a_list[ind]):
            tally = 0



def read_video_h5(name):
    hdf5_file = h5py.File('../data/%s/%s.h5' % (name, name), 'r')
    swapped_fullpath = list(np.genfromtxt('../data/%s/swapped_fullpath_names.txt' % name, dtype=None))
    og_path = list(np.genfromtxt('../data/%s/fullpath_sequence_names.txt' % name, dtype=None))

    for i in range(10):
        path = og_path[i]
        all_images = os.listdir(path)
        all_images.sort()
        path = os.path.join(path, all_images[0])

        sequence = hdf5_file[swapped_fullpath[i]][:]
        image_from_h5 = sequence[0].astype('uint8')
        # image_from_file = ndimage.imread('/home/gabi/Documents/datasets/ilids-vid-fixed/cam_a/person_0001/sequence_000/000.png')
        image_from_file = ndimage.imread(path)

        plt.imshow(image_from_h5)
        plt.imshow(image_from_file)


def download_from_internet():
    raw_data_path = '../raw_data'
    if not os.path.exists(raw_data_path):
        os.mkdir(raw_data_path)
    # download the zip file
    urllib.urlretrieve('http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip', os.path.join(raw_data_path, 'viper.zip'))
    # extract
    zip_ref = zipfile.ZipFile(os.path.join(raw_data_path, 'viper.zip'), 'r')
    zip_ref.extractall('../raw-data')


def parent():
    folder_path = '/home/gabi/Documents/datasets/CUHK/CUHK1'
    print(os.path.dirname(folder_path))


def get_image_from_h5():
    path_h5 = '../data/GRID/grid.h5'
    h5thing = h5py.File(path_h5, 'r')
    keys = list(np.genfromtxt('/home/gabi/PycharmProjects/uatu/data/GRID/swapped_list_of_paths.txt', dtype=None))

    num = 10

    for item in keys:
        if num != 0:
            print(item)
            thingy = h5thing[item][:]
            print(np.shape(thingy), type(thingy))
            num -= 1
        else:
            break


get_image_from_h5()