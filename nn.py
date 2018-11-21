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

'''COMMENTING MFCC STUFF FOR NOW

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


MFCCs Commented'''


##################################____WAV DATA ANALYSIS BELOW______####################

print("###############################BEGINNING WAV ANALYSIS#########################################")
print("##############################################################################################")
print("##############################################################################################")
print("##############################################################################################")
print("##############################################################################################")
print("##############################################################################################")
print("")
print("")

def load_wavs(wav_folder, split_num):
    files = listdir_ignore_hidden(wav_folder)
    wav_arr = []
    arr_len = 0
    for file in files:
        file_path = os.path.join(wav_folder, file)
        audio_seg = AudioSegment.from_file(file_path)
        array = audio_seg.get_array_of_samples()
        array = array[::split_num]
        if arr_len != 0:
            if arr_len == len(array):
                wav_arr.append(array)
        else:
            arr_len = len(array)
            wav_arr.append(array)
    return wav_arr


na_wavs = load_wavs(path.dirname(__file__) + "/Audio_Data/Wav_Data/North_America", 1)
ind_wavs = load_wavs(path.dirname(__file__) + "/Audio_Data/Wav_Data/India", 1)





def wav_create_array_and_labels(*argv):
    num_classes = len(argv)
    non_empty_arrays = True
    data_array = []
    label_array = []
    while(non_empty_arrays):
        non_empty_arrays = False
        arr_choice = numpy.random.randint(0,num_classes)
        #print(arr_choice)
        if(len(argv[arr_choice]) != 0):
            #print(str(len(argv[arr_choice])) + ": " + str(arr_choice))
            data_array.append(numpy.asarray(argv[arr_choice].pop()))
            label_array.append(arr_choice)
        for arg in argv:
            if len(arg) != 0:
                non_empty_arrays = True
    data_array = numpy.asarray(data_array)
    data_array.reshape(data_array.shape[0], data_array.shape[1], 1)
    label_array = numpy.asarray(label_array)
    label_array = keras.utils.to_categorical(label_array, num_classes)
    return (data_array, label_array)




wav_data_array, wav_label_array = wav_create_array_and_labels(na_wavs, ind_wavs)

#print(wav_data_array[0])
#print(type(wav_data_array[0]))
#print(wav_label_array[0])
#print(wav_data_array.shape)



wav_train_data, wav_train_labels, wav_test_data, wav_test_labels = uniform_random_sampling(wav_data_array, wav_label_array, 1500)

print(wav_train_data.shape)
print(wav_train_data[0][0])
print(wav_train_data[0][1])
print(wav_train_data[0][2])
#print(wav_train_labels.shape)
#print(wav_test_data.shape)
#print(wav_test_labels.shape)


'''
#This is the one that pretty much came straight from an example
#Compiles but gives very bad classification rate
wav_model = Sequential()
wav_model.add(Dense(12, input_dim=22050, activation='relu'))
wav_model.add(Dense(8, activation='relu'))
wav_model.add(Dense(2, activation='sigmoid'))

wav_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

wav_model.fit(wav_train_data, wav_train_labels, epochs=15, batch_size=10, validation_data=(wav_test_data, wav_test_labels))

score = wav_model.evaluate(wav_test_data, wav_test_labels, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''




'''
#1D CNN Model
#Compiles but also gives very bad classification return_state
#Maybe this can be tuned to give a better accuracy?

#Had to reshape the data a little bit for this one
wav_train_data_R = numpy.expand_dims(wav_train_data, axis=2)
print("New shape: " + str(wav_train_data.shape))
wav_test_data_R = numpy.expand_dims(wav_test_data, axis=2)

#Second Wav Model
wav_model = Sequential()
#wav_model.add(Dense(12, input_dim=22050, activation='relu'))
#wav_model.add(Dense(8, activation='relu'))
#wav_model.add(Dense(2, activation='sigmoid'))
wav_model.add(Conv1D(32, 10, activation='relu', input_shape=(wav_train_data_R.shape[1],1)))
wav_model.add(MaxPooling1D(pool_size=2))
wav_model.add(Conv1D(32, 10))
wav_model.add(MaxPooling1D(pool_size=2))
wav_model.add(Flatten())
wav_model.add(Dropout(0.25))
wav_model.add(Dense(128, activation='relu'))
wav_model.add(Dropout(0.5))
wav_model.add(Dense(2, activation='sigmoid'))

wav_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

wav_model.fit(wav_train_data_R, wav_train_labels, epochs=15, batch_size=10, validation_data=(wav_test_data_R, wav_test_labels))

score = wav_model.evaluate(wav_test_data_R, wav_test_labels, verbose=0)

print('NEW Test loss:', score[0])
print('NEW Test accuracy:', score[1])

'''



'''

#LSTM MODEL
#Not sure what's up with this....
#This runs very slowly and overflowed my memory when I tried to run it
wav_train_data_R = numpy.expand_dims(wav_train_data, axis=2)
print("New shape: " + str(wav_train_data_R.shape))
wav_test_data_R = numpy.expand_dims(wav_test_data, axis=2)


rnn_model = Sequential()
rnn_model.add(LSTM(1000, input_dim=1, dropout=0.25, recurrent_dropout=0.25, return_sequences=True, input_shape=(wav_train_data_R.shape[1],1)))
#rnn_model.add(Dense(2,activation='sigmoid'))

rnn_model.add(LSTM(32,return_sequences=False))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(output_dim=2, activation='sigmoid'))


rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.fit(wav_train_data_R, wav_train_labels, epochs=3, batch_size=64)


score = rnn_model.evaluate(wav_test_data_R, wav_test_labels, verbose=0)

print('NEW Test loss:', score[0])
print('NEW Test accuracy:', score[1])


'''


'''
#WORK IN PROGRESS
#Couldn't get this one to run properly
wav_train_data_R = numpy.expand_dims(wav_train_data, axis=2)
print("New shape: " + str(wav_train_data_R.shape))
wav_test_data_R = numpy.expand_dims(wav_test_data, axis=2)

model = Sequential()
model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

print ('Compiling...')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


model.fit(wav_train_data_R, wav_train_labels, epochs=5, batch_size=1000, validation_data=(wav_test_data_R, wav_test_labels))

score = model.evaluate(wav_test_data_R, wav_test_labels, verbose=0)



print(model.summary())

'''





'''

#CONVINCING MYSELF THAT SAMPLING METHODS WORK
#THIS LETS YOU FOLLOW THE TRACE OF THE ARRAYS THROUGH THE SAMPLING

testArr1 = []
testArr1.append(array.array('h', [1, 11, 111]))
testArr1.append(array.array('h', [1111, 1111, 1111]))
testArr1.append(array.array('h', [333, 333, 333]))
testArr2 = []
testArr2.append(array.array('h', [2,22,222]))
testArr2.append(array.array('h', [2222, 2222, 2222]))
testArr2.append(array.array('h', [444, 444, 444]))
testArr2.append(array.array('h', [4444, 4444, 4444]))


test_data_array, test_label_array = wav_create_array_and_labels(testArr1, testArr2)

print(test_data_array)
print(test_label_array)


test_train_data, test_train_labels, test_test_data, test_test_labels = uniform_random_sampling(test_data_array, test_label_array, 4)

print("")
print("Test_Train")
print(test_train_data)
print(test_train_labels)
print("")
print("Test_Test")
print(test_test_data)
print(test_test_labels)



'''





#THIS IS MY CURRENT ATTEMPT
#SWITCHING AROUND SOME CODE FOR LSTM RNN'S I FOUND ONLINE


input_file = '/Users/spencerriggins/Pattern_Recognition/RNN_Test/input.csv'

def load_data(test_split = 0.2):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    df['sequence'] = df['sequence'].apply(lambda x: [int(e) for e in x.split()])
    df = df.reindex(numpy.random.permutation(df.index))

    train_size = int(len(df) * (1 - test_split))

    X_train = df['sequence'].values[:train_size]
    y_train = numpy.array(df['target'].values[:train_size])
    X_test = numpy.array(df['sequence'].values[train_size:])
    y_test = numpy.array(df['target'].values[train_size:])

    return pad_sequences(X_train), y_train, pad_sequences(X_test), y_test




def create_model(input):
    print ('Creating model...')
    model = Sequential()
    #Removed Embedding Layer
    model.add(LSTM(batch_input_shape = input.shape, units = 100, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    #model.add(Dropout(0.5))
    #model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    #model.add(Dropout(0.5))
    #model.add(Dense(2, activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


X_train, y_train, X_test, y_test = load_data()

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

#print(X_train.shape)
#print(y_train.shape)

print("OLDSHAPES")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



wav_train_data_R = numpy.expand_dims(wav_train_data, axis=2)
print("New shape: " + str(wav_train_data_R.shape))
wav_test_data_R = numpy.expand_dims(wav_test_data, axis=2)


X_train = wav_train_data_R
y_train = wav_train_labels
X_test = wav_test_data_R
y_test = wav_test_labels

#print(X_train[0][0][0])


print("")
print("NEWSHAPES")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)





model = create_model(X_train)

print ('Fitting model...')
hist = model.fit(X_train, y_train, batch_size=1500, epochs=3, validation_split = 0.1, verbose = 1)

score, acc = model.evaluate(X_test, y_test, batch_size=1)
print('Test score:', score)
print('Test accuracy:', acc)

print(X_test)
print(y_test)

print(model.predict(X_test, verbose=1))
