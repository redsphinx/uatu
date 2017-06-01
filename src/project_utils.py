"""
Useful utilities
"""

import numpy as np
import project_constants as pc
import os
from shutil import copyfile
import shutil
import time
from random import shuffle

# recursively transform list into tuple
def tupconv(lst):
    tuplst = []
    for x in lst:
        if isinstance(x, np.ndarray):
            tuplst.append(tupconv(x))
        elif isinstance(x, list):
            tuplst.append(tupconv(x))
        else:
            tuplst.append(x)
    return tuple(tuplst)


# used to calculate error
def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def get_wrong_predictions():
    folder = 'wrong_predictions'

    paths = np.genfromtxt('test_images.csv', dtype=None).tolist()
    ans = np.genfromtxt('wrong_predictions.txt', dtype=None).tolist()

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.mkdir(folder)

    for line in range(0, len(ans)):
        step = ans[line].split(',')[1]
        if step == 'testing':
            target = ans[line].split(',')[3]
            prediction = ans[line].split(',')[5]
            if not target == prediction:
                thing = paths[line].split('/')[-1]
                copyfile(paths[line], os.path.join(folder, thing))

# todo: debug
def threshold_predictions(adjustable, predictions):
    num_pred = len(predictions)
    if adjustable.cost_module_type == 'neural_network':
        new_predictions = np.zeros((num_pred, 2))
        for item in range(num_pred):
            new_predictions[item, np.argmax(predictions[item])] = 1

        return new_predictions
    elif adjustable.cost_module_type == 'euclidean':
        predictions = predictions.ravel()
        new_predictions = [0] * num_pred
        for item in range(num_pred):
            if predictions[item] < adjustable.distance_threshold:
                new_predictions[item] = 0
            else:
                new_predictions[item] = 1

        new_predictions = np.asarray(new_predictions)
        return new_predictions



def calculate_accuracy(predictions, labels):
    predictions = threshold_predictions(predictions)
    good = 0.0
    total = len(predictions) * 1.0

    if len(np.shape(labels)) > 1:
        for pred in range(0, len(predictions)):
            if predictions[pred][0] == labels[pred][0]:
                good += 1
    else:
        for pred in range(0, len(predictions)):
            if predictions[pred] == labels[pred]:
                good += 1

    acc = good / total
    return acc


# FIXME make it euclidean distance compatible
def make_confusion_matrix(adjustable, predictions, labels):
    predictions = threshold_predictions(adjustable, predictions)
    tp, fp, tn, fn = 0, 0, 0, 0
    if adjustable.cost_module_type == 'neural_network':
        for lab in range(0, len(labels)):
            if labels[lab][0] == 0:
                if predictions[lab][0] == 0:
                    tp += 1
                else:
                    fn += 1
            elif labels[lab][0] == 1:
                if predictions[lab][0] == 1:
                    tn += 1
                else:
                    fp += 1
    elif adjustable.cost_module_type == 'euclidean':
        for lab in range(0, len(labels)):
            if labels[lab] == 0:
                if predictions[lab] == 0:
                    tp += 1  # t=1, p=1
                else:
                    fn += 1  # t=1, p=0
            elif labels[lab] == 1:
                if predictions[lab] == 1:
                    tn += 1
                else:
                    fp += 1

    return [tp, fp, tn, fn]


def enter_in_log(experiment_name, file_name, data_names, matrix_means, matrix_std, ranking_means, ranking_std, total_time):
    decimals = '.2f'
    if not os.path.exists(pc.LOG_FILE_PATH):
        with open(pc.LOG_FILE_PATH, 'w') as my_file:
            print('new log file made')

    with open(pc.LOG_FILE_PATH, 'a') as log_file:
        date = str(time.strftime("%d/%m/%Y")) + "   " + str(time.strftime("%H:%M:%S"))
        log_file.write('\n')
        log_file.write('name_of_experiment:         %s\n' % experiment_name)
        log_file.write('file_name:                  %s\n' % file_name)
        log_file.write('date:                       %s\n' % date)
        log_file.write('duration:                   %f\n' % total_time)
        for dataset in range(len(data_names)):
            name = data_names[dataset]

            log_file.write('%s mean tp,fp,tn,fn:    %s\n' % (name, str(reduce_float_length(np.asarray(matrix_means[dataset]).tolist(), decimals))))
            log_file.write('%s std tp,fp,tn,fn:     %s\n' % (name, str(reduce_float_length(np.asarray(matrix_std[dataset]).tolist(), decimals))))
            log_file.write('%s mean ranking:        %s\n' % (name, str(reduce_float_length(np.asarray(ranking_means[dataset]).tolist(), decimals))))
            log_file.write('%s std ranking:         %s\n' % (name, str(reduce_float_length(np.asarray(ranking_std[dataset]).tolist(), decimals))))
        log_file.write('\n')


# FIXME make comptible with euclidean distance
def calculate_CMC(adjustable, predictions):
    # TODO: debug
    ble = np.shape(predictions)

    if adjustable.cost_module_type == 'neural_network':
        predictions = np.reshape(predictions[:, 1], (pc.RANKING_NUMBER, pc.RANKING_NUMBER))
    elif adjustable.cost_module_type == 'euclidean':
        predictions = predictions.ravel()
        predictions = np.reshape(predictions, (pc.RANKING_NUMBER, pc.RANKING_NUMBER))
    else:
        predictions = None

    ranking_matrix_abs = np.zeros((pc.RANKING_NUMBER, pc.RANKING_NUMBER))
    tallies = np.zeros(pc.RANKING_NUMBER)
    final_ranking = []

    for row in range(len(predictions)):
        # get the indices by sorted values from high to low
        ranking_matrix_abs[row] = [i[0] for i in sorted(enumerate(predictions[row]), key=lambda x: x[1],
                                                        reverse=True)]
        list_form = ranking_matrix_abs[row].tolist()
        num = list_form.index(row)
        tallies[num] += 1

    for tally in range(len(tallies)):
        percentage = sum(tallies[0:tally + 1]) * 1.0 / sum(tallies) * 1.0
        final_ranking.append(float('%0.2f' % percentage))
    return final_ranking


def reduce_float_length(a_list, decimals):
    for i in range(len(a_list)):
        a_list[i] = float(format(a_list[i], decimals))
    return a_list


def sideways_shuffle(data_list):
    """ Data comes in already shuffled but only horizontally. I think that the order matters because the features get
        concatenated. And in `combinations` the first item gets paired with others while always being in the left column
        So we take half of the rows and swap the locations of item1, item2. Labels don't change. And then we shuffle the
        list again.
    """
    cutoff = len(data_list) / 2
    to_be_shuffled = data_list[0:cutoff]

    column_1 = [item.strip().split(',')[0] for item in to_be_shuffled]
    column_2 = [item.strip().split(',')[1] for item in to_be_shuffled]
    labels = [item.strip().split(',')[-1] for item in to_be_shuffled]

    shuffled_list = [column_2[i] + ',' + column_1[i] + ',' + labels[i] for i in range(cutoff)]

    data_list[0:cutoff] = shuffled_list

    shuffle(data_list)

    return data_list


def flip_labels(data_list):
    """ Gets a list of pairs and labels and flips the labels. so 1 becomes 0 and 0 becomes 1
    """
    column_1 = [item.strip().split(',')[0] for item in data_list]
    column_2 = [item.strip().split(',')[1] for item in data_list]
    labels = [item.strip().split(',')[-1] for item in data_list]
    labels = list(np.asarray(np.asarray([1]*len(labels)) - np.asarray(labels, dtype=int), dtype=str))

    data_list = [column_1[i] + ',' + column_2[i] + ',' + labels[i] for i in range(len(column_1))]

    return data_list


# def assign_experiments():
#     import running_experiments as re
#     list_of_experiments = []
#     list_of_experiments = ['experishit']
#     # for i in range(57, 69 + 1):
#     #     list_of_experiments.append('experiment_%d' % i)
#     number_of_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"])
#     for gpu in range(number_of_gpus):
#         if gpu_in_use(gpu) == False:
#             the_experiment = getattr(re, list_of_experiments.pop(0))
#             the_experiment(gpu)
#
# assign_experiments()

