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

# dynamic variables, depends on the person: image width, height and number of frames.
# determine these when loading the data

def make_balanced_dataset():
    labels_ = DATA_DIRECTORY + '/pair_labels.csv'
    labels = pd.read_csv(labels_)
    labels = labels.sort_values('class')
    labels_ = labels.to_csv('labels_sorted.csv', index=False)
    with open('labels_sorted.csv') as f:
        reader = csv.reader(f)
        labels = list(reader)
    # print(len(labels))
    # print(labels[0])
    # print(labels[-1])
    # print(labels[len(labels)-10:-1])
    labels_neg = labels[1:len(labels)-200]
    rd.shuffle(labels_neg)
    balanced_labels = labels_neg[0:200] + labels[len(labels)-201:-1]

def main():
    # extract data into numpy arrays
    pass


make_balanced_dataset()