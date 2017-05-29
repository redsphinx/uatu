from keras.models import load_model
import numpy as np
import dynamic_data_loading as ddl
import project_constants as pc
import project_data_handling as pd
import project_utils as pu
import os
import keras
from scipy import ndimage
from PIL import Image
from skimage.util import random_noise
from matplotlib.image import imsave
from itertools import combinations



def zoom(image):
    the_image = image
    image_width, image_height = the_image.size
    image_2 = the_image.crop((image_width*0.15, image_height*0.15, image_width*0.85, image_height*0.85))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2


def rotate(image):
    the_image = image
    image_2 = the_image.rotate(20)
    image_width, image_height = the_image.size
    image_2 = image_2.crop((image_width * 0.15, image_height * 0.15, image_width * 0.85, image_height * 0.85))
    image_2 = image_2.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
    return image_2

def noise(path):
    the_image = ndimage.imread(path)
    image_2 = random_noise(the_image)
    return image_2


def flip(image):
    the_image = image
    image_2 = the_image.transpose(Image.FLIP_LEFT_RIGHT)
    return image_2


def create_and_save_augmented_images(keys, the_id):
    # augments data. saves data
    path = '../data/CUHK02/augmented/%s' % the_id
    if not os.path.exists(path): os.mkdir(path)

    for item in keys:
        image = Image.open(item)
        name_bare = item.strip().split('/')[-1].split('.')[0]
        name_original = os.path.join(path, name_bare + '_original.png')
        image.save(name_original)

        image_zoom = zoom(image)
        name_zoom = os.path.join(path, name_bare + '_zoom.png')
        image_zoom.save(name_zoom)

        image_rotate = rotate(image)
        name_rotate = os.path.join(path, name_bare + '_rotate.png')
        image_rotate.save(name_rotate)

        image_noise = noise(item)
        name_noise = os.path.join(path, name_bare + '_noise.png')
        imsave(name_noise, image_noise)

        image_vertical_flip = flip(image)
        name_flip = os.path.join(path, name_bare + '_flip.png')
        image_vertical_flip.save(name_flip)


def load_augmented_images(list_augmented_images):
    combos = combinations(list_augmented_images, 2)

    len_combos = 0
    for item in combos: len_combos += 1

    data = np.zeros((len_combos, 2, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, 3))

    i = 0
    for comb in combos:
        image_1 = ndimage.imread(comb[0])
        image_2 = ndimage.imread(comb[1])
        data[i][0] = image_1
        data[i][1] = image_2
        i += 1

    return data


def main(adjustable):

    cuhk02_ranking = list(np.genfromtxt('cuhk02_ranking.txt', dtype=None))
    market_ranking = list(np.genfromtxt('market_ranking.txt', dtype=None))

    all_ranking = [cuhk02_ranking, market_ranking]
    names = ['cuhk02', 'market']

    confusion_matrices = []
    ranking_matrices = []

    for item in range(len(all_ranking)):
        name = names[item]
        this_ranking = all_ranking[item]

        if name == 'cuhk02':
            full_predictions = []
            h5_dataset = ddl.load_datasets_from_h5(['cuhk02'])

            for id in range(pc.RANKING_NUMBER):
                matching_index = id * pc.RANKING_NUMBER + id
                partition = this_ranking[matching_index].strip().split(',')[0].split('+')[-3]
                image_1 = this_ranking[matching_index].strip().split(',')[0].split('+')[-1]
                image_2 = this_ranking[matching_index].strip().split(',')[1].split('+')[-1]
                seen = [image_1, image_2]
                the_id = pd.my_join(list(image_1)[0:5])


                list_related_keys = ddl.get_related_keys(name, partition, the_id, seen)
                path = '../data/CUHK02/augmented/%s' % the_id
                if not os.path.exists(path):
                    create_and_save_augmented_images(list_related_keys, the_id)

                list_augmented_images = os.listdir(path)
                list_augmented_images = [os.path.join(path, item) for item in list_augmented_images]

                # training
                prime_train = load_augmented_images(list_augmented_images)
                training_instances = len(prime_train[:, 0])
                prime_labels = np.ones(training_instances, dtype=int)
                prime_labels = keras.utils.to_categorical(prime_labels, pc.NUM_CLASSES)

                model = load_model('scn_86_model.h5')
                model.fit([prime_train[0], prime_train[:, 1]], prime_labels,
                          batch_size=adjustable.batch_size,
                          epochs=adjustable.prime_epochs,
                          verbose=2)

                # testing
                part_ranking = this_ranking[id * pc.RANKING_NUMBER:(id + 1) * pc.RANKING_NUMBER]
                test_data = ddl.grab_em_by_the_keys(part_ranking, h5_dataset)

                part_prediction = model.predict([test_data[0, :], test_data[1, :]])
                full_predictions += part_prediction


            # putting it all together
            final_testing_labels = [int(this_ranking[item].strip().split(',')[-1]) for item in
                                    range(len(this_ranking))]
            final_testing_labels = keras.utils.to_categorical(final_testing_labels, pc.NUM_CLASSES)

            matrix = pu.make_confusion_matrix(full_predictions, final_testing_labels)
            accuracy = (matrix[0] + matrix[2]) * 1.0 / (sum(matrix) * 1.0)
            if not matrix[0] == 0:
                precision = (matrix[0] * 1.0 / (matrix[0] + matrix[1] * 1.0))
            else:
                precision = 0
            confusion_matrices.append(matrix)

            ranking = pu.calculate_CMC(full_predictions)
            ranking_matrices.append(ranking)

            print('%s accuracy: %0.2f   precision: %0.2f   confusion matrix: %s \n CMC: \n %s'
                  % (name, accuracy, precision, str(matrix), str(ranking)))

        else:
            pass

