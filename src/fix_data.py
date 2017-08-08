"""
Author:     Gabrielle Ras
E-mail:     flambuyan@gmail.com

File contains methods to fix the raw datasets to a format that the neural network accepts
"""

import os
from PIL import Image
from shutil import copyfile
import project_constants as pc
import project_utils as pu


def standardize(all_images, folder_path, fixed_folder_path, the_mod=None):
    """
    Assuming the images have been downloaded and extracted.
    Makes the images in the CAVIAR4REID, GRID and PRID450 in the correct size of 128x64
    :param all_images:          list of image names
    :param folder_path:         string, the directory of the extracted images
    :param fixed_folder_path:   string, the directory of the standardized images
    :param the_mod:             string, modifier to add to the image name, so `image_a` where `the_mod = '_a'`
    """

    def modify(name, a_mod):
        return name.split('.')[0].split('_')[-1] + a_mod + name.split('.')[-1]

    for image in all_images:
        original_image_path = os.path.join(folder_path, image)

        if the_mod is not None:
            modified_image_path = os.path.join(fixed_folder_path, modify(image, the_mod))
        else:
            modified_image_path = os.path.join(fixed_folder_path, image)
        the_image = Image.open(original_image_path)
        image_width, image_height = the_image.size

        if image_width < pc.IMAGE_WIDTH and image_height < pc.IMAGE_HEIGHT:
            case = 1
        elif image_width > pc.IMAGE_WIDTH and image_height > pc.IMAGE_HEIGHT:
            case = 2

        elif image_width < pc.IMAGE_WIDTH and image_height > pc.IMAGE_HEIGHT:
            case = 3
        elif image_width > pc.IMAGE_WIDTH and image_height < pc.IMAGE_HEIGHT:
            case = 4

        elif image_width < pc.IMAGE_WIDTH and image_height == pc.IMAGE_HEIGHT:
            case = 1
        elif image_width > pc.IMAGE_WIDTH and image_height == pc.IMAGE_HEIGHT:
            case = 2

        elif image_width == pc.IMAGE_WIDTH and image_height > pc.IMAGE_HEIGHT:
            case = 2
        elif image_width == pc.IMAGE_WIDTH and image_height < pc.IMAGE_HEIGHT:
            case = 1

        elif image_width == pc.IMAGE_WIDTH and image_height == pc.IMAGE_HEIGHT:
            case = 5
        else:
            case = None

        # if dimensions are bigger than WIDTH, HEIGHT then resize
        # if dimensions are smaller then pad with zeros
        if case == 2:
            the_image = the_image.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
            the_image.save(modified_image_path)
        elif case == 1 or case == 3 or case == 4:
            if case == 3:
                the_image = the_image.resize((image_width, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
            elif case == 4:
                the_image = the_image.resize((pc.IMAGE_WIDTH, image_height), Image.ANTIALIAS)
            padding_width = (pc.IMAGE_WIDTH - the_image.size[0]) / 2
            padding_height = (pc.IMAGE_HEIGHT - the_image.size[1]) / 2
            new_img = Image.new('RGB', (pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), (255, 255, 255))
            new_img.paste(the_image, box=(padding_width, padding_height))
            new_img.save(modified_image_path)
        elif case == 5:
            the_image.save(modified_image_path)


def fix_viper():
    """
    Assuming the VIPeR data is already downloaded and extracted, put all images in a single folder and pad the
    width with zeros.
    """
    original_folder_path = '/home/gabi/Documents/datasets/VIPeR'
    padded_folder_path = '/home/gabi/Documents/datasets/VIPeR/padded'

    if not os.path.exists(padded_folder_path):
        os.mkdir(padded_folder_path)

    cams = ['cam_a', 'cam_b']
    for folder in cams:
        cam_path = os.path.join(original_folder_path, str(folder))
        padded_cam_path = padded_folder_path
        num = 'a' if folder == 'cam_a' else 'b'
        for the_file in os.listdir(cam_path):
            img = Image.open(os.path.join(cam_path, the_file))
            new_img = Image.new('RGB', (pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), (255, 255, 255))

            img_width, img_height = img.size
            new_img_width, new_img_height = new_img.size
            padding_width = (new_img_width-img_width)/2
            padding_height = (new_img_height-img_height)/2

            new_img.paste(img, box=(padding_width, padding_height))

            filename = the_file.split('_')[0] + '_' + str(num) + '.bmp'
            filename = os.path.join(padded_cam_path, filename)
            new_img.save(filename)


def fix_cuhk01():
    """
    Assuming the CUHK01 data is already downloaded and extracted, put all images in a single folder and resizes
    the images from 160x60 to 128x64.
    """
    folder_path = '/home/gabi/Documents/datasets/CUHK/CUHK1'
    new_path = os.path.dirname(folder_path)

    list_images = os.listdir(folder_path)
    name_folder = folder_path.split('/')[-1]
    new_folder_path = os.path.join(new_path, 'cropped_' + str(name_folder))
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    for image_path in list_images:
        img = Image.open(os.path.join(folder_path, image_path))
        img = img.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
        img.save(os.path.join(new_folder_path, image_path))


def fix_cuhk02():
    """
    Assuming the CUHK02 data is already downloaded and extracted, put all images in a single folder and resizes
    the images from 160x60 to 128x64. Notice the weird layout of the folder structure. We leave the dataset partitioned
    in 5 parts.
    """
    folder_path = '/home/gabi/Documents/datasets/CUHK/CUHK2'
    cropped_folder_path = os.path.join(os.path.dirname(folder_path), 'cropped_CUHK2')
    if not os.path.exists(cropped_folder_path):
        os.mkdir(cropped_folder_path)

    subdirs = os.listdir(folder_path)

    for a_dir in subdirs:
        if not os.path.exists(os.path.join(cropped_folder_path, a_dir)):
            os.makedirs(os.path.join(cropped_folder_path, a_dir, 'all'))

    cameras = ['cam1', 'cam2']

    for a_dir in subdirs:
        for cam in cameras:
            original_images_path = os.path.join(folder_path, a_dir, cam)
            cropped_images_path = os.path.join(cropped_folder_path, a_dir, 'all')
            images = [a_file for a_file in os.listdir(original_images_path) if a_file.endswith('.png')]
            for ind in range(len(images)):
                image = os.path.join(original_images_path, images[ind])
                cropped_image = os.path.join(cropped_images_path, images[ind])
                img = Image.open(image)
                img = img.resize((pc.IMAGE_WIDTH, pc.IMAGE_HEIGHT), Image.ANTIALIAS)
                img.save(cropped_image)


def fix_caviar():
    """
    Assuming the CAVIAR4REID data is already downloaded and extracted, standardizes images to 128x64.
    """
    folder_path = '/home/gabi/Documents/datasets/CAVIAR4REID/original'
    fixed_folder_path = os.path.join(os.path.dirname(folder_path), 'fixed_caviar')
    if not os.path.exists(fixed_folder_path):
        os.mkdir(fixed_folder_path)

    all_images = os.listdir(folder_path)
    standardize(all_images, folder_path, fixed_folder_path)


def fix_grid():
    """
    Assuming the GRID data is already downloaded and extracted, standardizes images to 128x64.
    """
    folder_path = '/home/gabi/Documents/datasets/GRID'
    probe = os.path.join(folder_path, 'probe')
    gallery = os.path.join(folder_path, 'gallery')

    probe_list = os.listdir(probe)
    gallery_list = os.listdir(gallery)

    # trim the gallery list and remove items with '0000' in the path, these are identities that do not belong to a pair
    proper_gallery_list = [item for item in gallery_list if not item[0:4] == '0000']

    fixed_folder_path = os.path.join(os.path.dirname(probe), 'fixed_grid')
    if not os.path.exists(fixed_folder_path):
        os.mkdir(fixed_folder_path)

    # standardize will put probe and gallery in the same fixed folder
    standardize(probe_list, probe, fixed_folder_path)
    standardize(proper_gallery_list, gallery, fixed_folder_path)


def fix_prid450():
    """
    Assuming the PRID450 data is already downloaded and extracted, standardizes images to 128x64.
    """
    folder_path = '/home/gabi/Documents/datasets/PRID450'
    cam_a = os.path.join(folder_path, 'cam_a')
    cam_b = os.path.join(folder_path, 'cam_b')

    cam_a_list = os.listdir(cam_a)
    cam_b_list = os.listdir(cam_b)

    # trim the dataset to contain only RGB color images
    proper_cam_a_list = [item for item in cam_a_list if item.split('_')[0] == 'img']
    proper_cam_b_list = [item for item in cam_b_list if item.split('_')[0] == 'img']

    fixed_folder_path = os.path.join(os.path.dirname(cam_a), 'fixed_prid')
    if not os.path.exists(fixed_folder_path):
        os.mkdir(fixed_folder_path)

    # standardize will put probe and gallery in the same fixed folder
    standardize(proper_cam_a_list, cam_a, fixed_folder_path, '_a.')
    standardize(proper_cam_b_list, cam_b, fixed_folder_path, '_b.')


def fix_video_dataset(name, min_seq):
    """
    Make all sequences of same length and store them in a new folder
    :param name:        name of the video dataset: 'ilids-vid' or 'prid2011'
    :param min_seq:     minimal sequence length
    """
    old_path = '/home/gabi/Documents/datasets/%s' % name

    # make new directory
    new_path = '/home/gabi/Documents/datasets/%s-fixed' % name
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    # get the cameras
    cams = os.listdir(old_path)

    for cam in cams:
        if cam == 'cam1':
            cam_new = 'cam_a'
        elif cam == 'cam2':
            cam_new = 'cam_b'
        else:
            cam_new = cam
        new_cam_path = os.path.join(new_path, cam_new)
        if not os.path.exists(new_cam_path):
            os.mkdir(new_cam_path)
        old_cam_path = os.path.join(old_path, cam)
        persons = os.listdir(old_cam_path)

        # list the persons
        for person in persons:
            if len(person.split('_')) == 1:
                new_person = list(person)[-3:]
                new_person = pu.my_join(new_person)
                new_person = int(new_person)
                new_person = 'person_%04d' % new_person
            else:
                new_person = person

            old_person_path = os.path.join(old_cam_path, person)
            images = sorted(os.listdir(old_person_path))

            # number of images in sequence
            number_images = len(images)

            # only continue if sequence has more than min_seq number of frames
            if number_images >= min_seq:
                new_person_path = os.path.join(new_cam_path, new_person)
                if not os.path.exists(new_person_path):
                    os.mkdir(new_person_path)

                possible_sequence_cuts = number_images / min_seq

                # depending on how many cuts we can make
                for sequence in range(possible_sequence_cuts):
                    sequence_path = os.path.join(new_person_path, 'sequence_%03d' % sequence)
                    if not os.path.exists(sequence_path):
                        os.mkdir(sequence_path)

                    sample_images = images[sequence * min_seq: min_seq + sequence * min_seq]
                    number_sample_images = len(sample_images)

                    for s_i in range(number_sample_images):
                        old_image = os.path.join(old_person_path, sample_images[s_i])
                        name_s_i = '%03d.png' % s_i
                        new_image = os.path.join(sequence_path, name_s_i)

                        # copy file
                        copyfile(old_image, new_image)
            else:
                print(old_person_path, number_images)


def fix_prid2011():
    """ turn into
        prid2011-fixed / cam_x / person_xxx / sequence_xxx / xxx.png
    """
    fix_video_dataset('prid2011', 20)


def fix_ilids():
    """ turn into
        ilids-vid-fixed / cam_x / person_xxx / sequence_xxx / xxx.png
    """
    fix_video_dataset('ilids-vid', 22)


def fix_inria():
    """
    Crops INRIA dataset images around center point and saves in new folder.
    Assuming the dataset is already downloaded.
    Since we are using these images to pre-train we will use all the images, so no specific testing/training split
    """
    # 2416 images
    path_train_positive = '/home/gabi/Documents/datasets/INRIAPerson/train_64x128_H96/pos'
    # 1218 images
    path_train_negative = '/home/gabi/Documents/datasets/INRIAPerson/train_64x128_H96/neg'
    # 1132 images
    path_test_positive = '/home/gabi/Documents/datasets/INRIAPerson/test_64x128_H96/pos'
    # 453 images
    path_test_negative = '/home/gabi/Documents/datasets/INRIAPerson/test_64x128_H96/neg'

    parent_path = '/home/gabi/Documents/datasets/INRIAPerson'
    fixed_positive = os.path.join(parent_path, 'fixed-pos')
    fixed_negative = os.path.join(parent_path, 'fixed-neg')

    if not os.path.exists(fixed_positive):
        os.mkdir(fixed_positive)
    if not os.path.exists(fixed_negative):
        os.mkdir(fixed_negative)

    # for positive images, we crop around the center
    for path in [path_train_positive, path_test_positive]:
        list_images = os.listdir(path)
        for image in list_images:
            old_image_path = os.path.join(path, image)
            new_image_path = os.path.join(fixed_positive, image)
            image_data = Image.open(old_image_path)
            width, height = image_data.size

            center_x = width / 2
            center_y = height / 2
            crop_x = center_x - pc.IMAGE_WIDTH / 2
            crop_y = center_y - pc.IMAGE_HEIGHT / 2

            cropped_image = image_data.crop((crop_x, crop_y, crop_x + pc.IMAGE_WIDTH, crop_y + pc.IMAGE_HEIGHT))
            cropped_image.save(new_image_path)

    # for negative images, we crop and do data augmentation by flipping the image horizontally
    for path in [path_train_negative, path_test_negative]:
        list_images = os.listdir(path)
        for image in list_images:
            old_image_path = os.path.join(path, image)
            new_image_path = os.path.join(fixed_negative, image)
            image_bare, suffix = image.split('.')
            flipped_path = os.path.join(fixed_negative, image_bare + '_flipped.' + suffix)
            image_data = Image.open(old_image_path)
            width, height = image_data.size

            center_x = width / 2
            center_y = height / 2
            crop_x = center_x - pc.IMAGE_WIDTH / 2
            crop_y = center_y - pc.IMAGE_HEIGHT / 2

            cropped_image = image_data.crop((crop_x, crop_y, crop_x + pc.IMAGE_WIDTH, crop_y + pc.IMAGE_HEIGHT))
            cropped_image.save(new_image_path)

            flipped_image = cropped_image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_image.save(flipped_path)
