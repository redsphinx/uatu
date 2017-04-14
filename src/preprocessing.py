import numpy as np
import project_constants as pc
import project_utils as pu
from PIL import Image
import os
import random as rd
from scipy import ndimage
from shutil import copyfile
import shutil
from itertools import combinations
import random
import csv
import time


# do this operation after image has been loaded into memory
def center(data):
    print('centering the data by subtracting the mean per channel')
    data_shape = np.shape(data)
    centered_data = np.zeros(data_shape)
    mean_data = np.zeros((3, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH))
    # calculate mean per channel of entire dataset
    for channel in range(data_shape[-1]):
        mean_data[channel] = np.mean(data[:,:,:,channel], axis=0)

    # for each image in dataset
    for image in range(data_shape[0]):
        # for each channel subtract mean
        for channel in range(data_shape[-1]):
            centered_data[image, :, :, channel] = data[image,:,:,channel] - mean_data[channel]

    return centered_data


def normalize(data):
    print('normalizing the data by dividing by the std')
    data_shape = np.shape(data)
    normalized_data = np.zeros(data_shape)
    std_data = np.zeros((3, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH))
    # calculate std per channel of entire dataset
    for channel in range(data_shape[-1]):
        std_data[channel] = np.std(data[:, :, :, channel], axis=0)

    # for each image in dataset
    for image in range(data_shape[0]):
        # for each channel divide by std
        for channel in range(data_shape[-1]):
            normalized_data[image, :, :, channel] = data[image, :, :, channel] / std_data[channel]

    return normalized_data


def PCA_whiten(data):
    print('PCA whitening the data: take dataset with subtracted mean, calculate the covariance matrix, '
          'decorrelate the data, divide by eigenvalues')
    data_shape = np.shape(data)
    whitened_data = np.zeros((data_shape[0], data_shape[1], data_shape[2], data_shape[-1]))
    cov_data = np.zeros((data_shape[-1], data_shape[2], data_shape[2]))
    # because we take the dot product
    rot_data = np.zeros((data_shape[-1], data_shape[1], data_shape[2]))

    # for each image in dataset
    for image in range(data_shape[0]):
        # for each channel divide by std
        for channel in range(data_shape[-1]):
            cov_data[channel] = np.dot(np.transpose(data[image,:,:,channel]), data[image,:,:,channel]) / data_shape[0]
            u, s, v = np.linalg.svd(cov_data[channel])
            rot_data[channel] = np.dot(data[image, :, :, channel], u)
            whitened_data[image, :, :, channel] = rot_data[channel] / np.sqrt(s + 1e-5)

    return whitened_data