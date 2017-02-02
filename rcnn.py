import tensorflow as tf
import os
import numpy as np
import pandas as pd
import csv
import random as rd

rd.seed(42)

DATA_DIRECTORY = 'PRID2011'
NUMBER_OF_TRACKLETS = len(os.listdir(DATA_DIRECTORY)) - 1
VALIDATION_SIZE = 50
NUM_LABELS = 2


# make the dataset balanced
def balance_data():
    labels_ = DATA_DIRECTORY + '/pair_labels.csv'
    labels = pd.read_csv(labels_)
    labels = labels.sort_values('class')
    labels.to_csv('labels_sorted.csv', index=False)
    # now labels are sorted with class 0 on top and 1 at the bottom
    with open('labels_sorted.csv') as f:
        reader = csv.reader(f)
        labels = list(reader)
    labels_neg = labels[1:len(labels)-200]
    rd.shuffle(labels_neg)
    labels_pos = labels[len(labels)-201:-1]
    rd.shuffle(labels_pos)
    balanced_labels = labels_neg[0:200] + labels_pos
    rd.shuffle(balanced_labels)
    return balanced_labels


# create the validation dataset
def create_data():
    labels = balance_data()
    print(labels[-1])
    train = labels[0:len(labels)-VALIDATION_SIZE]
    validation = labels[len(labels)-VALIDATION_SIZE:len(labels)]
    return [train, validation]




def main():
    # extract data into numpy arrays


    pass

create_data()