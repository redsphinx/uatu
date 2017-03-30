import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
import project_constants as pc
import project_utils as pu
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def base_model(input_dim):
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    return seq


