# a file for making test methods to investigate whatever

import tensorflow as tf
import keras
from keras import models
import os
# import keras
# from keras.models import Sequential, Model
# from keras.layers import Dense, Dropout, Activation, LSTM, Embedding, Input
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from scipy import io
import scipy
from scipy.misc import imread
import project_constants as pc

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
    print(keras.__version__)


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


def test_load_mat_2():
    path = '/home/gabi/Documents/datasets/mpii-human-pose/mpii_human_pose_v1_u12_1.mat'
    the_file = io.loadmat(path)

    release = the_file['RELEASE']
    single_people = release['single_person']

    print(type(the_file), type(release), type(single_people))

    print('asdf')


def test_get_h5_data():
    path = '../data/INRIA/inria.h5'
    h5data = h5py.File(path, 'r')

    key = '+home+gabi+Documents+datasets+INRIAPerson+fixed-pos+crop001812d.png'

    print(h5data[key][:])


def get_image_names_from_mat():
    path = '/home/gabi/Documents/datasets/mpii-human-pose/mpii_human_pose_v1_u12_1.mat'
    the_file = io.loadmat(path)
    release = the_file['RELEASE']
    annolist = release['annolist'][0, 0]
    image_names = annolist['image'][0]

    data_folder = '../data/mpii'
    mapping_name_file = os.path.join(data_folder, 'mapping_name.txt')

    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    len_data = len(image_names)

    with open(mapping_name_file, 'w') as my_file:
        for item in range(len_data):
            name = str(image_names[item][0][0][0][0])
            line = '%s,%s\n' % (item, name)
            my_file.write(line)


def get_rectangle_id_for_single_human():
    path = '/home/gabi/Documents/datasets/mpii-human-pose/mpii_human_pose_v1_u12_1.mat'
    the_file = io.loadmat(path)
    release = the_file['RELEASE']
    persons = release['single_person'][0, 0]

    data_folder = '../data/mpii'
    mapping_rid_file = os.path.join(data_folder, 'mapping_rid.txt')

    len_data = len(persons)

    with open(mapping_rid_file, 'w') as my_file:
        for item in range(len_data):
            person = persons[item][0]
            rids = range(len(person))
            rids.append(len(person))
            rids.pop(0)
            line = '%s,%s\n' % (item, str(rids))
            my_file.write(line)


def get_rid_xy_for_single_human():
    path = '/home/gabi/Documents/datasets/mpii-human-pose/mpii_human_pose_v1_u12_1.mat'
    the_file = io.loadmat(path)
    release = the_file['RELEASE']
    annolist = release['annolist'][0, 0]
    annorect = annolist['annorect'][0]

    len_data = len(annorect)

    for item in range(100):
        a3 = annorect[item][0][0]

        a3_len = len(a3)

        if a3_len == 2:
            xy = a3[1]
            x = int(xy[0][0][0])
            y = int(xy[0][0][1])
            print(item, 'x', x, 'y', y)

        elif a3_len > 2:
            pass
            # for i in range(a3_len):
            #     print(a3[i])

        asdf = 'asdf'

    # data_folder = '../data/mpii'
    # mapping_name_file = os.path.join(data_folder, 'mapping_name.txt')
    #
    # if not os.path.exists(data_folder):
    #     os.mkdir(data_folder)
    #
    # len_data = len(image_names)
    #
    # with open(mapping_name_file, 'w') as my_file:
    #     for item in range(len_data):
    #         name = str(image_names[item][0][0][0][0])
    #         line = '%s,%s\n' % (item, name)
    #         my_file.write(line)

# get_rid_xy_for_single_human()

def look_weights():
    path = '/home/gabi/PycharmProjects/uatu/model_weights/scnn_26072017_1834_epoch_100_weigths.h5'
    h5file = h5py.File(path, 'r')

    stuff = h5file['sequential_1']

    for i in range(len(stuff.keys())):
        k = stuff.keys()[i]
        print(k)
        p = 'sequential_1' + '/' + k
        for ii in range(len(h5file[p].keys())):
            print(h5file[p].keys()[ii])


    print('shit')


def compare_image():

    img_1_path = '/home/gabi/Documents/datasets/GRID/fixed_grid/0001_1_25004_107_32_106_221.jpeg'
    img_2_path = '/home/gabi/Documents/datasets/GRID/fixed_grid/0001_2_25023_116_134_128_330.jpeg'
    img_3_path = '/home/gabi/Documents/datasets/GRID/fixed_grid/0002_1_25008_169_19_94_224.jpeg'


    pair = np.zeros((1, 2, 128, 64, 3))

    img_1 = imread(img_1_path)
    img_2 = imread(img_2_path)
    img_3 = imread(img_3_path)


    pair[0, 0] = img_1
    print(np.shape(pair[0, 0]))
    pair[0, 1] = img_2


    model = models.load_model('/home/gabi/PycharmProjects/uatu/model_weights/priming_on_viper_epoch_100_model.h5')

    prediction = model.predict([pair[:, 0], pair[:, 1]])
    print(prediction)
    pair[0, 1] = img_3
    prediction = model.predict([pair[:, 0], pair[:, 1]])
    print(prediction)


def dataset_mean():
    # gets mean of a dataset and saves it

    # Access all PNG files in directory
    # allfiles = os.listdir(os.getcwd())
    the_path = '//home/gabi/PycharmProjects/uatu/src/dataset_averages/cuhk02'
    allfiles = os.listdir(the_path)
    imlist = [os.path.join(the_path, item) for item in allfiles]

    # imlist = [filename for filename in allfiles if filename[-4:] in [".png", ".PNG"]]
    # imlist = [filename for filename in allfiles if filename[-5:] in [".jpeg"]]

    # Assuming all images are the same size, get dimensions of first image
    w, h = Image.open(imlist[0]).size
    N = len(imlist)

    # Create a nparray of floats to store the average (assume RGB images)
    arr = np.zeros((h, w, 3), np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr = np.array(Image.open(im), dtype=np.float)
        arr = arr + imarr / N

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)

    # Generate, save and preview final image
    out = Image.fromarray(arr, mode="RGB")
    out.save("Average_cuhk02.png")
    out.show()

dataset_mean()

def find_matches(file_1, file_2):
    f1 = list(np.genfromtxt(file_1, dtype=None))
    f2 = list(np.genfromtxt(file_2, dtype=None))

    f1_id = [item.strip().split('/')[-1].split('_')[-1] for item in f1]
    f2_id = [item.strip().split('/')[-1].split('_')[-1] for item in f2]

    inters = set(f1_id).intersection(f2_id)

    print('intersection: ', len(inters))

    f1_diff = set(f1_id) - set(inters)
    f2_diff = set(f2_id) - set(inters)
    
    count_f1 = 0
    for item in f1_diff:
        index_f1_id = f1_id.index(item)
        f1_path = f1[index_f1_id]
        num_of_images = len(os.listdir(f1_path))
        if num_of_images > 40:
            count_f1 += 1
    print('count f1: ', count_f1)

    count_f2 = 0
    for item in f2_diff:
        index_f2_id = f2_id.index(item)
        f2_path = f2[index_f2_id]
        num_of_images = len(os.listdir(f2_path))
        if num_of_images > 40:
            count_f2 += 1
    print('count f2: ', count_f2)

    print('total: ', len(inters) + count_f1 + count_f2)


def zoom(image):
    the_image = image
    image_2 = the_image.crop((5, 5, 59, 123))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2


def rotate(image):
    the_image = image
    image_2 = the_image.rotate(4)
    image_2 = image_2.crop((5, 5, 59, 123))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2


def flip_zoom(image):
    the_image = image
    image_2 = the_image.transpose(Image.FLIP_LEFT_RIGHT)
    image_2 = image_2.crop((5, 5, 59, 123))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2


def flip_rotate(image):
    the_image = image
    image_2 = the_image.transpose(Image.FLIP_LEFT_RIGHT)
    image_2 = image_2.rotate(4)
    image_2 = image_2.crop((5, 5, 59, 123))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2


def test_augmentation():
    image_path = '/home/gabi/Documents/datasets/GRID/fixed_grid/0009_2_25226_176_72_87_246.jpeg'
    image = Image.open(image_path)
    plt.imshow(image)

    image_zoom = zoom(image)
    plt.imshow(image_zoom)

    image_rotate = rotate(image)
    plt.imshow(image_rotate)

    image_flip_zoom = flip_zoom(image)
    plt.imshow(image_flip_zoom)

    image_flip_rotate = flip_rotate(image)
    plt.imshow(image_flip_rotate)
