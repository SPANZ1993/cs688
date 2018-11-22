from keras.models import Sequential
from keras.layers import Dense
from scipy import misc
import convert_WAVtoMFCC
import numpy
from pydub import AudioSegment
from FeatureExtraction import listdir_ignore_hidden
from os import path
import os
import imageio
from FeatureExtraction import create_mfcc_array
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, RNN, SimpleRNNCell, Embedding
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
from keras import backend as K
import copy
import time
import array
import pandas as pd

numpy.random.seed(7)


'''
#CANT REMEMBER WHAT THIS IS BUT DON'T WORRY ABOUT IT
#I DON'T THINK IT IS IMPORTANT
if K.image_data_format() == 'channels_first':
    print("channels_first")
    #x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    #input_shape = (1, img_rows, img_cols)
else:
    print("not_channels_first")
    #x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    #input_shape = (img_rows, img_cols, 1)
'''



###########################__SAMPLING_METHODS_USED_BY_BOTH__#########################


def uniform_random_sampling(data, labels, sample_size):
    assert(len(data) == len(labels))
    complement_data = copy.deepcopy(data)
    complement_labels = copy.deepcopy(labels)
    data_size = len(complement_data)
    #print("((((((((()))))))))")
    #print(complement_data.shape)
    #print(complement_labels.shape)
    #print("((((((((()))))))))")
    sampled_data_shape = [sample_size]
    count = 0
    for s in complement_data.shape:
        if(count!=0):
            sampled_data_shape.append(s)
        count = count+1
    sampled_data_shape = tuple(sampled_data_shape)
    sampled_labels_shape = [sample_size]
    count = 0
    for s in complement_labels.shape:
        if(count!=0):
            sampled_labels_shape.append(s)
        count = count + 1
    sampled_labels_shape = tuple(sampled_labels_shape)
    #print("WOO")
    #print(sampled_data_shape)
    #print(sampled_labels_shape)
    sampled_data = numpy.zeros(shape=(sampled_data_shape))
    sampled_labels = numpy.zeros(shape=sampled_labels_shape)
    for i in range(0, sample_size):
        arr_choice = numpy.random.randint(0,data_size-i)
        #sampled_data.append(complement_data[arr_choice])
        sampled_data[i] = complement_data[arr_choice]
        complement_data = numpy.delete(complement_data, arr_choice, 0)
        #sampled_labels.append(complement_labels[arr_choice])
        sampled_labels[i] = complement_labels[arr_choice]
        complement_labels = numpy.delete(complement_labels, arr_choice, 0)
    #sampled_data = numpy.asarray(sampled_data)
    #complement_data = numpy.asarray(complement_data)
    #sampled_labels = numpy.asarray(complement_data)
    #complement_labels = numpy.asarray(complement_labels)
    return (sampled_data, sampled_labels, complement_data, complement_labels)



def sample_for_k_folds(data, labels, num_folds):
    assert(len(data) == len(labels))
    complement_data = copy.deepcopy(data)
    complement_labels = copy.deepcopy(labels)
    print("IN K FOLDS SAMPLE")
    print("LENGTH: " + str(len(data)))
    sample_size = len(data)//num_folds
    print("SAMPLE SIZE: " + str(sample_size))
    folds_arr = []
    for fold in range(num_folds):
        sampled_data, sampled_labels, complement_data, complement_labels = uniform_random_sampling(complement_data, complement_labels, sample_size)
        folds_arr.append([sampled_data, sampled_labels])
    return folds_arr




###########################__MFCC_ANALYIS_BELOW__#########################


def load_mfccs(mfcc_folder):
    files = listdir_ignore_hidden(mfcc_folder)
    mfcc_arr = []
    for file in files:
        file_path = os.path.join(mfcc_folder, file)
        #mfcc = misc.imread(file_path)
        mfcc = imageio.imread(file_path)
        mfcc_arr.append(mfcc)
        #print(mfcc)
    return mfcc_arr
na_mfcc_arr = create_mfcc_array(path.dirname(__file__) + "/Audio_Data/Wav_Data/North_America")
ind_mfcc_arr = create_mfcc_array(path.dirname(__file__) + "/Audio_Data/Wav_Data/India")
#0 Label Corresponds to North America
#1 Label Corresponds to India
def create_array_and_labels(*argv):
    num_classes = len(argv)
    #print("NUM:")
    #print(num_classes)
    time.sleep(5)
    non_empty_arrays = True
    data_array = []
    label_array = []
    while(non_empty_arrays):
        non_empty_arrays = False
        arr_choice = numpy.random.randint(0,num_classes)
        #print(arr_choice)
        if(len(argv[arr_choice]) != 0):
            data_array.append(argv[arr_choice].pop())
            label_array.append(arr_choice)
        for arg in argv:
            if len(arg) != 0:
                non_empty_arrays = True
    data_array = numpy.asarray(data_array)
    data_array = data_array.reshape(data_array.shape[0], data_array.shape[1], data_array.shape[2], 1)
    label_array = numpy.asarray(label_array)
    label_array = keras.utils.to_categorical(label_array, num_classes)
    return (data_array, label_array)
#MAKE SURE THIS WORKS CORRECTLY
def sample_for_k_folds(data, labels, num_folds):
    assert(len(data) == len(labels))
    complement_data = copy.deepcopy(data)
    complement_labels = copy.deepcopy(labels)
    print("IN K FOLDS SAMPLE")
    print("LENGTH: " + str(len(data)))
    sample_size = len(data)//num_folds
    print("SAMPLE SIZE: " + str(sample_size))
    folds_arr = []
    for fold in range(num_folds):
        sampled_data, sampled_labels, complement_data, complement_labels = uniform_random_sampling(complement_data, complement_labels, sample_size)
        folds_arr.append([sampled_data, sampled_labels])
    return folds_arr
data_arr, label_arr = create_array_and_labels(na_mfcc_arr, ind_mfcc_arr)
#folds_arr = sample_for_k_folds(data_arr, label_arr, 5)
#print(data_arr.shape)
#print(label_arr.shape)
sampled_train_data, sampled_train_labels, sampled_test_data, sampled_test_labels = uniform_random_sampling(data_arr, label_arr, data_arr.shape[0] - 200)
#print(sampled_train_data.shape)
#print(sampled_train_labels.shape)
#print(sampled_test_data.shape)
#print(sampled_test_labels.shape)
print("BEGINNING MFCC TRAINING.....")
input_shape = (sampled_train_data.shape[1],sampled_train_data.shape[2],sampled_train_data.shape[3])
num_classes = 2
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
batch_size = 100
model.fit(sampled_train_data, sampled_train_labels,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(sampled_test_data, sampled_test_labels))
score = model.evaluate(sampled_test_data, sampled_test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
