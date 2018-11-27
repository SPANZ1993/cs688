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
from FeatureExtraction import create_mfcc_array, create_chroma_array, create_feature_dictionary
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


na_feature_dict = create_feature_dictionary(path.dirname(__file__) + "/Audio_Data/Wav_Data/North_America")
ind_feature_dict = create_feature_dictionary(path.dirname(__file__) + "/Audio_Data/Wav_Data/India")

for i in range(5):
    print("MFCC" + str(i))
    for j in range(5):
        print(na_feature_dict['MFCC'][i][j])
    print("Chroma" + str(i))
    for k in range(5):
        print(na_feature_dict['Chroma'][i][k])



print("NA_DICT")
print(len(na_feature_dict['MFCC']))
print(len(na_feature_dict['Chroma']))
print("IND_DICT")
print(len(ind_feature_dict['MFCC']))
print(len(ind_feature_dict['Chroma']))


def create_array_and_labels_for_cotraining(*argv):
    num_classes = len(argv)
    #print("NUM:")
    #print(num_classes)
    time.sleep(5)
    non_empty_arrays = True
    mfcc_data_array = []
    chroma_data_array = []
    label_array = []
    count = 0
    while(non_empty_arrays):
        #print("ROUND: " + str(count))
        non_empty_arrays = False
        arr_choice = numpy.random.randint(0,num_classes)
        #print(arr_choice)
        #if(len(argv[arr_choice]) != 0):
        if (len(argv[arr_choice]['MFCC']) != 0) and (len(argv[arr_choice]['MFCC']) == len(argv[arr_choice]['Chroma'])):
            mfcc_data_array.append(argv[arr_choice]['MFCC'].pop())
            chroma_data_array.append(argv[arr_choice]['Chroma'].pop())
            label_array.append(arr_choice)
        for arg in argv:
            if (len(arg['MFCC']) != 0) or (len(arg['Chroma']) != 0):
                non_empty_arrays = True
                if len(arg['MFCC']) != len(arg['Chroma']):
                    print("PROBLEM")
                    print("DIFFERENT LENGTHS")
    #for i in range(5):
#        print(data_array[i])
#        print(type(data_array[i]))
#        print(data_array[i].shape)
#        print(type(data_array))
        count = count + 1
    mfcc_data_array = numpy.asarray(mfcc_data_array)
    chroma_data_array = numpy.asarray(chroma_data_array)



    #print(data_array.shape)
    #for i in range(5):
        #print(data_array[i].shape)
    #ADDED ADDITIONAL FORMATTING (ONLY EXECUTES ON CHROMA DATA)
    if(len(chroma_data_array.shape) != len(chroma_data_array[0].shape)+1):
        print("TRIGGERED")
        print("TRIGGERED")
        new_shape = (chroma_data_array.shape[0],) + chroma_data_array[0].shape
        #print("NEW")
        #print(new_shape)
        new_list = []
        for i in range(chroma_data_array.shape[0]):
            #if(i<5):
                #print(i)
            for j in chroma_data_array[i].flatten().tolist():
                new_list.append(j)
        new_list = numpy.asarray(new_list)
        chroma_data_array = new_list.reshape(new_shape)

    mfcc_data_array = mfcc_data_array.reshape(mfcc_data_array.shape[0], mfcc_data_array.shape[1], mfcc_data_array.shape[2], 1)
    chroma_data_array = chroma_data_array.reshape(chroma_data_array.shape[0], chroma_data_array.shape[1], chroma_data_array.shape[2], 1)
    label_array = numpy.asarray(label_array)
    label_array = keras.utils.to_categorical(label_array, num_classes)
    return (mfcc_data_array, chroma_data_array, label_array)



mfcc_data_array, chroma_data_array, label_array = create_array_and_labels_for_cotraining(na_feature_dict, ind_feature_dict)




print(mfcc_data_array.shape)
print(chroma_data_array.shape)
print(label_array.shape)



def uniform_random_sampling_for_cotraining(mfcc_data, chroma_data, labels, sample_size):
    assert(len(mfcc_data) == len(labels) and len(mfcc_data) == len(chroma_data))
    complement_mfcc_data = copy.deepcopy(mfcc_data)
    complement_chroma_data = copy.deepcopy(chroma_data)
    complement_labels = copy.deepcopy(labels)
    data_size = len(complement_mfcc_data)
    #print("((((((((()))))))))")
    #print(complement_data.shape)
    #print(complement_labels.shape)
    #print("((((((((()))))))))")


    #Determine Shape of sampled MFCC data
    sampled_mfcc_data_shape = [sample_size]
    count = 0
    for s in complement_mfcc_data.shape:
        if(count!=0):
            sampled_mfcc_data_shape.append(s)
        count = count+1
    sampled_mfcc_data_shape = tuple(sampled_mfcc_data_shape)



    #Determine Shape of sampled Chroma data
    sampled_chroma_data_shape = [sample_size]
    count = 0
    for s in complement_chroma_data.shape:
        if(count!=0):
            sampled_chroma_data_shape.append(s)
        count = count+1
    sampled_chroma_data_shape = tuple(sampled_chroma_data_shape)


    #Determine Shape of sampled Chroma data
    #Should end up being (sample_size, n_classes)
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
    sampled_mfcc_data = numpy.zeros(shape=(sampled_mfcc_data_shape))
    sampled_chroma_data = numpy.zeros(shape=(sampled_chroma_data_shape))
    sampled_labels = numpy.zeros(shape=sampled_labels_shape)
    for i in range(0, sample_size):
        arr_choice = numpy.random.randint(0,data_size-i)
        #sampled_data.append(complement_data[arr_choice])
        sampled_mfcc_data[i] = complement_mfcc_data[arr_choice]
        sampled_chroma_data[i] = complement_chroma_data[arr_choice]

        complement_mfcc_data = numpy.delete(complement_mfcc_data, arr_choice, 0)
        complement_chroma_data = numpy.delete(complement_chroma_data, arr_choice, 0)

        #sampled_labels.append(complement_labels[arr_choice])
        sampled_labels[i] = complement_labels[arr_choice]
        complement_labels = numpy.delete(complement_labels, arr_choice, 0)
    #sampled_data = numpy.asarray(sampled_data)
    #complement_data = numpy.asarray(complement_data)
    #sampled_labels = numpy.asarray(complement_data)
    #complement_labels = numpy.asarray(complement_labels)
    return (sampled_mfcc_data, sampled_chroma_data, sampled_labels, complement_mfcc_data, complement_chroma_data, complement_labels)


smfcc_data, schroma_data, s_labels, cmfcc_data, cchroma_data, c_labels = uniform_random_sampling_for_cotraining(mfcc_data_array, chroma_data_array, label_array, label_array.shape[0]-200)

print("SAMPLED")
print(smfcc_data.shape)
print(schroma_data.shape)
print(s_labels.shape)
print("Complement")
print(cmfcc_data.shape)
print(cchroma_data.shape)
print(c_labels.shape)


'''
#FIRST TRAINING ON MFCC
print("FIRST TRAINING ON MFCC..")
print("")

sampled_train_data = smfcc_data
sampled_train_labels = s_labels
sampled_test_data = cmfcc_data
sampled_test_labels = c_labels


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

print("MFCC:")
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''

'''
#This model can accept MFCC and Chroma data
print("TRAINING MODEL")
print("")

sampled_train_data = schroma_data
sampled_train_labels = s_labels
sampled_test_data = cchroma_data
sampled_test_labels = c_labels


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

print("Chroma:")
print('Test loss:', score[0])
print('Test accuracy:', score[1])


predictions = model.predict(sampled_train_data)
for i in range(3):
    print(predictions[i])
print(type(predictions[0]))
'''




###################################COTRAINING BELOW####################################

print("##############################__BEGIN_COTRAINING__###############################")
print("#################################################################################")
print("#################################################################################")
print("#################################################################################")


########################################


na_feature_dict = create_feature_dictionary(path.dirname(__file__) + "/Audio_Data/Wav_Data/North_America")
ind_feature_dict = create_feature_dictionary(path.dirname(__file__) + "/Audio_Data/Wav_Data/India")

mfcc_data_array, chroma_data_array, label_array = create_array_and_labels_for_cotraining(na_feature_dict, ind_feature_dict)


######

L_size = 200
Test_size = 200
U_mfcc_data, U_chroma_data, U_labels, L_mfcc_data, L_chroma_data, L_labels = uniform_random_sampling_for_cotraining(mfcc_data_array, chroma_data_array, label_array, label_array.shape[0]-L_size)
U_mfcc_data, U_chroma_data, U_labels, Test_mfcc_data, Test_chroma_data, Test_labels = uniform_random_sampling_for_cotraining(U_mfcc_data, U_chroma_data, U_labels, U_labels.shape[0]-Test_size)

U_Dict = {'MFCC': U_mfcc_data, 'Chroma': U_chroma_data, 'Labels': U_labels}
L_Dict = {'MFCC': L_mfcc_data, 'Chroma': L_chroma_data, 'Labels': L_labels} #L_Size number of samples
Test_Dict = {'MFCC': Test_mfcc_data, 'Chroma': Test_chroma_data, 'Labels': Test_labels} #Test_size number of samples


print("")
print("U_Dict")
print(U_Dict['MFCC'].shape)
print(U_Dict['Chroma'].shape)
print(U_Dict['Labels'].shape)
print("L_Dict")
print(L_Dict['MFCC'].shape)
print(L_Dict['Chroma'].shape)
print(L_Dict['Labels'].shape)
print("Test_Dict")
print(Test_Dict['MFCC'].shape)
print(Test_Dict['Chroma'].shape)
print(Test_Dict['Labels'].shape)


#####
####################h1 and h2###################

sampled_train_data = U_Dict['MFCC']
sampled_train_labels = U_Dict['Labels']
test_data = Test_Dict['MFCC']

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

h1 = copy.deepcopy(model)########################

#model.fit(L_Dict['MFCC'], L_Dict['Labels'], batch_size=10, verbose=1, epochs = 10)

print("H1___!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#print(model.predict(U_Dict['MFCC']))
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


#########################################


sampled_train_data = U_Dict['Chroma']
sampled_train_labels = U_Dict['Labels']
test_data = Test_Dict['Chroma']

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


h2 = copy.deepcopy(model)#####################

#model.fit(sampled_train_data, sampled_train_labels,
#          batch_size=batch_size,
#          epochs=10,
#          verbose=1)


print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#print(model.predict(test_data))
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


####################h1 and h2###################
#









def find_p_and_n_indices(p, n, predictions):
    print("Start")
    p_index_arr = [] #Indexes of p most likely positive samples
    n_index_arr = [] #Indexes of n most likely negative samples
    p_prob_arr = [] #Probabilities associated with current found indices
    n_prob_arr = []
    for i in range(p):
        p_prob_arr.append(float(0))
        p_index_arr.append(-1)
    for i in range(n):
        n_prob_arr.append(float(0))
        n_index_arr.append(-1)
    count = 0
    for i in range(len(predictions)):
        #print(count)
        #print(predictions[i])
        p_pred = predictions[i][0]
        n_pred = predictions[i][1]
        p_min_prob_index = p_prob_arr.index(min(p_prob_arr))
        n_min_prob_index = n_prob_arr.index(min(n_prob_arr))
        if(p_pred > p_prob_arr[p_min_prob_index]):
            p_prob_arr[p_min_prob_index] = p_pred
            p_index_arr[p_min_prob_index] = i
        if(n_pred > n_prob_arr[n_min_prob_index]):
            n_prob_arr[n_min_prob_index] = n_pred
            n_index_arr[n_min_prob_index] = i
        for nn in n_index_arr:
            for pp in p_index_arr:
                if nn==pp:
                    print("SHIZNIT WE FOUND THE SAME INDEX")
        count = count + 1
    return (p_index_arr, n_index_arr)


#Index arr:
#[[a,b,c], [d,e,f], [h,i]] -> Index a,b,c become labeled as 0; d,e,f become labeled as 1; h,i become labeled as 2
def add_indices_to_dict(index_arr, from_dict, to_dict):
    from_dict = copy.deepcopy(from_dict)
    to_dict = copy.deepcopy(to_dict)
    print("*************************************STARTING ADD INDICES*************************")
    print("Index_arr is")
    print(index_arr)
    print("from_dict shape")
    print(from_dict['MFCC'].shape)
    print(from_dict['Chroma'].shape)
    print(from_dict['Labels'].shape)
    print("to_dict shape")
    print(to_dict['MFCC'].shape)
    print(to_dict['Chroma'].shape)
    print(to_dict['Labels'].shape)
    temp_e_mfcc_arr = []
    temp_e_chroma_arr = []
    temp_e_label_arr = []
    for i in range(len(index_arr)):
        for index in index_arr[i]:
            e_mfcc = from_dict['MFCC'][index]
            print("e_mfcc")
            print(e_mfcc)
            print("e_mfcc initial shape:")
            print(e_mfcc.shape)
            e_mfcc_new_shape = [1]
            for i in e_mfcc.shape:
                e_mfcc_new_shape.append(i)

            e_mfcc_new_shape = tuple(e_mfcc_new_shape)
            e_mfcc.reshape(e_mfcc_new_shape)
            temp_e_mfcc_arr.append(e_mfcc)
            print("e_mfcc reshaped")
            print(e_mfcc)
            print("e_mfcc new shape")
            print(e_mfcc.shape)

            e_chroma = from_dict['Chroma'][index]
            print("e_chroma")
            print(e_chroma)
            print("e_chroma initial shape:")
            print(e_chroma.shape)
            e_chroma_new_shape = [1]
            for i in e_chroma.shape:
                e_chroma_new_shape.append(i)

            e_chroma_new_shape = tuple(e_chroma_new_shape)
            e_chroma.reshape(e_chroma_new_shape)
            temp_e_chroma_arr.append(e_chroma)
            print("e_chroma reshaped")
            print(e_chroma)
            print("e_chroma new shape")
            print(e_chroma.shape)

            e_label = keras.utils.to_categorical(i, len(index_arr))
            temp_e_label_arr.append(e_label)
            print(e_label)

    assert(len(temp_e_mfcc_arr) == len(temp_e_chroma_arr) == len(temp_e_label_arr))
    for i in range(len(temp_e_mfcc_arr)):
        e_mfcc = temp_e_mfcc_arr[i]
        e_chroma = temp_e_chroma_arr[i]
        e_label = temp_e_label_arr[i]
        to_dict['MFCC'] = numpy.append(to_dict['MFCC'], numpy.expand_dims(e_mfcc, axis = 0), axis = 0)
        to_dict['Chroma'] = numpy.append(to_dict['Chroma'], numpy.expand_dims(e_chroma, axis = 0), axis = 0)
        to_dict['Labels'] = numpy.append(to_dict['Labels'], numpy.expand_dims(e_label, axis = 0), axis = 0)


    #Just in case there are duplicates in the index_arr (There shouldn't be)
    #Removes duplicates in index_arr and returns it in reverse sorted order
    #to be deleted from from_dict
    found_nums = []
    for i in index_arr:
        for j in i:
            if(j not in found_nums):
                found_nums.append(j)
    found_nums.sort(reverse=True)
    index_arr = found_nums


    '''
    for i in range(len(index_arr)):
        j=0
        j_range = len(index_arr[i])
        while j < j_range:
            if(index_arr[i][j] in found_nums):
                found_nums.append(index_arr[i][j])
                index_arr[i].pop(j)
            else:
                found_nums.append(index_arr[i][j])
                j = j+1
            j_range = len(index_arr[i])
    '''


    for index in index_arr:
        from_dict['MFCC'] = numpy.delete(from_dict['MFCC'], index, 0)
        from_dict['Chroma'] = numpy.delete(from_dict['Chroma'], index, 0)
        from_dict['Labels'] = numpy.delete(from_dict['Labels'], index, 0)



    print("*************************************ENDING ADD INDICES*************************")
    print("Index_arr is")
    print(index_arr)
    print("from_dict shape")
    print(from_dict['MFCC'].shape)
    print(from_dict['Chroma'].shape)
    print(from_dict['Labels'].shape)
    print("to_dict shape")
    print(to_dict['MFCC'].shape)
    print(to_dict['Chroma'].shape)
    print(to_dict['Labels'].shape)

    return (from_dict, to_dict)



def replenish_dict(from_dict, to_dict, num_samples):
    if(from_dict['Labels'].shape[0] < num_samples): #In case from_dict does not have enough samples
        num_samples = from_dict['Labels'].shape
    indices = []
    while len(indices) < num_samples:
        rand_num = numpy.random.randint(0, from_dict['Labels'].shape[0])
        if(rand_num not in indices):
            indices.append(rand_num)
    temp_e_mfcc_arr = []
    temp_e_chroma_arr = []
    temp_e_label_arr = []
    for index in indices:
            e_mfcc = from_dict['MFCC'][index]
            temp_e_mfcc_arr.append(e_mfcc)
            e_chroma = from_dict['Chroma'][index]
            temp_e_chroma_arr.append(e_chroma)
            e_label = from_dict['Labels'][index]
            temp_e_label_arr.append(e_label)


    assert(len(temp_e_mfcc_arr) == len(temp_e_chroma_arr) == len(temp_e_label_arr))
    for i in range(len(temp_e_mfcc_arr)):
        e_mfcc = temp_e_mfcc_arr[i]
        e_chroma = temp_e_chroma_arr[i]
        e_label = temp_e_label_arr[i]
        to_dict['MFCC'] = numpy.append(to_dict['MFCC'], numpy.expand_dims(e_mfcc, axis = 0), axis = 0)
        to_dict['Chroma'] = numpy.append(to_dict['Chroma'], numpy.expand_dims(e_chroma, axis = 0), axis = 0)
        to_dict['Labels'] = numpy.append(to_dict['Labels'], numpy.expand_dims(e_label, axis = 0), axis = 0)

    indices.sort(reverse=True)

    for index in indices:
        from_dict['MFCC'] = numpy.delete(from_dict['MFCC'], index, 0)
        from_dict['Chroma'] = numpy.delete(from_dict['Chroma'], index, 0)
        from_dict['Labels'] = numpy.delete(from_dict['Labels'], index, 0)

    return (from_dict, to_dict)








p = 1
n = 3

k = 3
u = 200


#h1 -> classifier for feature 1
#h2 -> classifier for feature 2
#U -> Unlabeled Sample Data Set (Dictionary like {'MFCC': <MFCC training array>, 'Chroma': <Chroma training array>, 'Labels': <Label array>})
#Note: In U, Label array is only included for record-keeping, and is not used in training
#U_Prime -> Unlabeled Sample Data Pool (Dictionary like {'MFCC': <MFCC training array>, 'Chroma': <Chroma training array>, 'Labels': <Label array>})
#Note: U_Prime is constructed initialized by sampling u elements from U and then replenished by sampling 2p+2n samples from U per iteration
#L -> Labeled Training data (Dictionary like {'MFCC': <MFCC training array>, 'Chroma': <Chroma training array>, 'Labels': <Label array>})
#Note: L is initially small, then on each iteration both h1 and h2 choose p most likely positive and n most likely negative samples to add from
#U_prime into L
#k is the number of iterations co-training will run for
#test_data -> (Dictionary like {'MFCC': <MFCC training array>, 'Chroma': <Chroma training array>, 'Labels': <Label array>}) used for testing accuracy
def cotraining(h1, h2, U, L, p, n, k, u, test_data):
    #Save original settings of classifiers
    h1_copy = copy.deepcopy(h1)
    h2_copy = copy.deepcopy(h2)
    #FIX THESE
    smfcc_data, schroma_data, s_labels, cmfcc_data, cchroma_data, c_labels = uniform_random_sampling_for_cotraining(U['MFCC'], U['Chroma'], U['Labels'], U['Labels'].shape[0]-u)
    U = {'MFCC': smfcc_data, 'Chroma': schroma_data, 'Labels': s_labels}
    U_Prime = {'MFCC': cmfcc_data, 'Chroma': cchroma_data, 'Labels': c_labels}
    accuracy_arr = []
    h1_accuracy_arr = []
    h2_accuracy_arr = []
    for iteration in range(k):
        h1 = h1_copy
        h2 = h2_copy
        index_arr = []
        print("TRAINING H1")
        h1.fit(L['MFCC'], L['Labels'], batch_size=10, verbose=1, epochs=100, validation_data=(L['MFCC'], L['Labels']))
        print("ABOUT TO PREDICT")
        h1_predictions = h1.predict(U_Prime['MFCC']) #NEED TO MAKE THIS U_prime
        #h1_predictions = h1.predict(U['MFCC'])
        h1_score = h1.evaluate(test_data['MFCC'], test_data['Labels'], verbose=0)
        h1_accuracy_arr.append(h1_score[1])
        print("H1 Accuracy")
        print(h1_score[1])
        print("H1_Predictions")
        print(h1_predictions)
        print("JUST PREDICTED")
        h1_p, h1_n = find_p_and_n_indices(p, n, h1_predictions)
        print("H1 INDICES")
        for i in range(len(h1_p)):
            print("P" + str(i) + ": " + str(h1_p[i]))
        for i in range(len(h1_n)):
            print("N" + str(i) + ": " + str(h1_n[i]))

        print("TRAINING H2")
        h2.fit(L['Chroma'], L['Labels'], batch_size=10, verbose=1, epochs=10, validation_data=(L['Chroma'], L['Labels']))
        print("ABOUT TO PREDICT")
        h2_predictions = h2.predict(U_Prime['Chroma'])
        h2_score = h2.evaluate(test_data['Chroma'], test_data['Labels'], verbose=0)
        print("H2 Accuracy")
        print(h2_score[1])
        h2_accuracy_arr.append(h2_score[1])
        print("H2_Predictions")
        print(h2_predictions)
        print("JUST PREDICTED")
        h2_p, h2_n = find_p_and_n_indices(p, n, h2_predictions)
        assert(len(h1_p) == len(h2_p))
        assert(len(h1_n) == len(h2_n))
        print("H2 INDICES")
        for i in range(len(h2_p)):
            print("P" + str(i) + ": " + str(h2_p[i]))
        for i in range(len(h2_n)):
            print("N" + str(i) + ": " + str(h2_n[i]))

        p_index_arr = h1_p + h2_p
        n_index_arr = h1_n + h2_n
        index_arr = []
        index_arr.append(p_index_arr)
        index_arr.append(n_index_arr) #index_arr looks like [[p1,p2,p3,...], [n1,n2,n3,...]]
        U_Prime, L = add_indices_to_dict(index_arr, U_Prime, L)
        U, U_Prime = replenish_dict(U, U_Prime, (2*p)+(2*n))

    accuracy_arr.append(h1_accuracy_arr)
    accuracy_arr.append(h2_accuracy_arr)
    return accuracy_arr



'''
model.fit(sampled_train_data, sampled_train_labels,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(sampled_test_data, sampled_test_labels))
'''







cotraining_accuracy_arr = cotraining(h1, h2, U_Dict, L_Dict, p, n, k, u, Test_Dict)


#for i in range(10):
#    print(U_Dict['Chroma'][i])

print(cotraining_accuracy_arr)
