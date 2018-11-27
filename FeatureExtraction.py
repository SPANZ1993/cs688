from os import path, listdir
import numpy as np
import matplotlib.pyplot as plt
import os
import wave
import math
import statistics
import time
import re
import tensorflow as tf
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
from pydub.playback import play
from convert_WAVtoMFCC import convert_WAVtoMFCC, create_mfcc



#List all files in a directory except hidden files
#This does not include subdirectories
def listdir_ignore_hidden(path):
    files = []
    for file in os.listdir(path):
        #print(os.path.join(path,file))
        #if os.path.isdir(file):
            #print("FOUND A DIRECTORY")
        if not file.startswith('.') and not os.path.isdir(os.path.join(path,file)):
            files.append(file)
    #print("@@@@@@@@@@@")
    #for file in files:
        #print(file)
    #print("@@@@@@@@@@@")
    return files



#file_name = path.dirname(__file__) + "/Audio_Data/Mp3_Data/bengali6.mp3"
#output_file_name = path.dirname(__file__) + "/Audio_Data/Wav_Data/bengali6.wav"


'''
AudioSegment.from_mp3(file_name).export(output_file_name, format="wav") #FFMPEG PROBLEM



w = wave.open(path.dirname(__file__) + "/Audio_Data/Wav_Data/test.wav", 'r')


print(w.getnframes())
print(type(w))

signal = w.readframes(-1)
signal = np.fromstring(signal, 'Int16')


#If Stereo
if w.getnchannels() == 2:
    print('Just mono files')
    sys.exit(0)

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
#plt.show()
'''
#for i in range(1000):
    #print(signal[i], end = " ")

###############SPLIT ON SILENCE BELOW#######################


'''
#Find the ranges of frames that are counted as silent
sil_ranges = detect_silence(segW, min_silence_len = 250, silence_thresh = -28)
j = 0
for r in sil_ranges:
    print(str(r))
    j = j+1
    #play(segW[r[0]:r[1]])
print(str(j) + " Silent Ranges")
'''
'''
#Experimenting with manually creating word audio segments
word_ranges = []
for r in range(len(sil_ranges)):
    if r < len(sil_ranges) - 1:
        start = sil_ranges[r][1]
        end = sil_ranges[r+1][0]
    else:
        start = sil_ranges[r][1]
        end = sil_ranges[r][1] + 1
    #play(segW[start:end])
    word_ranges.append((start, end))

print("Word Range:")
print(word_ranges)

print(str(segW))
'''

'''
audio_files = listdir_ignore_hidden(path.dirname(__file__) + "/Audio_Data/Mp3_Data")

for file in audio_files:
    print(file)

    segW = AudioSegment.from_mp3(path.dirname(__file__) + "/Audio_Data/Mp3_Data/" + file)

    chunks = split_on_silence(segW, min_silence_len = 100, silence_thresh = -24, keep_silence = 200)

    n = 0
    for i, chunk in enumerate(chunks):
        n = n+1
        #time.sleep(0.5)
        #play(chunk)
        #print(type(chunk))
        #chunk.export(path.dirname(__file__) + "/Audio_Data/Wav_Data/chunk" + str(i+1) + ".wav")
        #if(i == 19):
        #        chunk.export(path.dirname(__file__) + "/Audio_Data/Wav_Data/wednesday.mp3", format = 'mp3')
    print(str(n) + " Chunks")
'''

from scipy.io.wavfile import read
'''
wavdata = (read(path.dirname(__file__) + "/Audio_Data/Wav_Data/LongIslandGirls.wav"))
print("HERE NOW")
print(type(wavdata))
print(str(wavdata))
wavdata = wavdata[1]
chunks2 = np.array_split(wavdata, 1)
print("AND ALSO HERE")
dbs = [20*math.log10( math.sqrt(statistics.mean(chunk**2)) ) for chunk in chunks2]
print("DBS:")
print(dbs)
'''

# [/\\]{1}([^/\\]+\w+)[\.]   <- REGEX FOR PARSING NAME OF FILE FROM FILE PATH

#INPUT: 1) Fully qualified name of an mp3 file 2) Fully qualified path where chunks will be stored as .wav files
#OUTPUT: Array of "chunks" of the audio in wav format saved in chunk_folder
def extract_chunks(mp3File):
    wavFolder = path.dirname(__file__) + "/Audio_Data/Wav_Data/"
    wavFile = convert_to_wav(mp3File, wavFolder) # <fileName>.mp3 --> <fileName>.wav
    avgDb = calculate_avg_db(wavFile)
    #print(type(avgDb))
    silence_thresh = -(0.8 * avgDb) #Threshold of decibels to count as silence
    #print(wavFile)
    segment = AudioSegment.from_wav(wavFile)
    #play(segment)
    #print("silence_thresh: " + str(silence_thresh))
    chunks = split_on_silence(segment, min_silence_len = 100, silence_thresh = silence_thresh, keep_silence = 200)
    file_name_base = extract_file_name(mp3File)
    return chunks


#INPUT: 1) An array of "chunks" (audio segments created from extract_chunks)
#OUTPUT: One audio segment created by combining all the chunks
def combine_chunks(chunks):
    combined_chunk = chunks[0]
    for chunk in chunks:
        if chunk != chunks[0]:
                combined_chunk = combined_chunk.append(chunk, crossfade = 0)
    return combined_chunk


#INPUT: 1) An audio segment 2) The length of each window in milliseconds
#OUTPUT: An array of "windows"
def extract_windows(audio, window_size):
    windows = audio[::window_size]
    window_arr = []
    for window in windows:
        if(len(window) == window_size):
            window_arr.append(window)
    return window_arr

#INPUT: 1) An array of windows (from extract_windows) 2) Fully qualified path name to save windows in 3) Name of file windows were generated from
#OUTPUT: Each window will be saved as a .wav file in the window_folder
# Ex: (windows, "/files/audiofiles/windows/", "test_windows") --> test_windows_0, test_windows_1, etc. in /windows/ directory
def save_windows(windows, window_folder, file_name):
    if not os.path.exists(window_folder):
        os.makedirs(window_folder)
    window_count = 0
    for window in windows:
        output_file_name = os.path.join(window_folder,file_name + "_" + str(window_count) + ".wav")
        #print("OUTPUT: " + output_file_name)
        #play(window)
        window.export(output_file_name, format="wav")
        window_count = window_count + 1
    #print("INNER COUNT: " + str(window_count))
    return



#INPUT: 1) Fully qualified name of mp3 file, 2) Fully qualified name of folder in which wav file will be created
#OUTPUT: A wav version of the file is created in the given output folder, and the path to this file is returned
#i.e. hello.mp3 -> hello.wav
def convert_to_wav(mp3File, output_folder):
    #file_name = os.path.basename(mp3File)
    #file_name = file_name.split('.')[0]
    file_name = extract_file_name(mp3File)
    #print(file_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #output_file_name = output_folder + file_name +".wav"
    output_file_name = os.path.join(output_folder,file_name + ".wav")
    AudioSegment.from_mp3(mp3File).export(output_file_name, format="wav")
    return output_file_name

#INPUT: 1) Fully qualified name of wav file
#OUTPUT: Average volume in dB of file
def calculate_avg_db(wavFile):
    wavdata = (read(wavFile))
    wavdata = wavdata[1]
    chunks = np.array_split(wavdata, 1)
    #for chunk in chunks:
    #    print(str(statistics.mean(chunk**2)))
    dbs = [20*math.log10( math.sqrt(statistics.mean(chunk**2)) ) for chunk in chunks]
    #print(dbs)
    return dbs[0]

#INPUT: Fully Qualified File name
#OUTPUT: Name of file
#Ex: /files/audio_files/audio_file.mp3 --> audio_file
def extract_file_name(file_name):
    file_name = os.path.basename(file_name)
    file_name = file_name.split('.')[0]
    return file_name


#INPUT: 1)Fully qualified path to folder with MP3 Files 2) Fully qualified path to where .wav windows will be saved 3) Desired length of each window (in ms)
#OUTPUT: All mp3 files in mp3_folder will be converted to windows and saved as .wav
def convert_folder_to_windows(mp3_folder, destination_folder, window_size):
    files = listdir_ignore_hidden(mp3_folder)
    for file in files:
        #print(file)
        file_name = extract_file_name(file)
        chunks = extract_chunks(os.path.join(mp3_folder,file))
        combined = combine_chunks(chunks)
        windows = extract_windows(combined, window_size)
        count = 0
        #for window in windows:
        #    count = count + 1
        #print("COUNT IS: " + str(count))
        save_windows(windows, destination_folder, file_name)
    return


#Input: Fully qualified path name to folder containing .wav files (window_folder), Fully qualified path name to new folder where MFCCs to be saved
#Output: Folder of mfcc data created from .wav files
def convert_folder_to_mfcc(wav_folder, mfcc_folder):
    files = listdir_ignore_hidden(wav_folder)
    for file in files:
        print(file)
        file_name = extract_file_name(file)
        convert_WAVtoMFCCWAVasMFCC(os.path.join(wav_folder,file), mfcc_folder)
    return


def create_mfcc_array(wav_folder):
    files = listdir_ignore_hidden(wav_folder)
    mfcc_arr = []
    print(len(files))
    count = 0
    for file in files:
        #print("Converting: " + str(count))
        count = count+1
        mfcc_arr.append(create_mfcc(os.path.join(wav_folder, file)))
        #print("WOOT")
        #print(mfcc_arr)
    print("Converted :" + str(len(files)))
    print("TYPE IS: " + str(type(mfcc_arr)))
    #mfcc_arr = np.asarray(mfcc_arr)
    #mfcc_arr = mfcc_arr.reshape(mfcc_arr.shape[0], mfcc_arr.shape[1], mfcc_arr.shape[2], 1)
    print("TYPE IS: " + str(type(mfcc_arr)))
    return mfcc_arr




#convert_folder_to_windows(path.dirname(__file__) + "/Audio_Data/Mp3_Data/North_America", path.dirname(__file__) + "/Audio_Data/Wav_Data/North_America", 500)
#convert_folder_to_windows(path.dirname(__file__) + "/Audio_Data/Mp3_Data/India", path.dirname(__file__) + "/Audio_Data/Wav_Data/India", 500)
#convert_folder_to_mfcc(path.dirname(__file__) + "/Audio_Data/Wav_Data/North_America", path.dirname(__file__) + "/Audio_Data/MFCC_Data/North_America")
#convert_folder_to_mfcc(path.dirname(__file__) + "/Audio_Data/Wav_Data/India", path.dirname(__file__) + "/Audio_Data/MFCC_Data/India")


#convert_folder_to_windows(path.dirname(__file__) + "/Audio_Data/Mp3_Data/Test_Hindi", path.dirname(__file__) + "/Audio_Data/Wav_Data/Test_Hindi", 500)
#convert_folder_to_mfcc(path.dirname(__file__) + "/Audio_Data/Wav_Data/Test_Hindi", path.dirname(__file__) + "/Audio_Data/MFCC_Data/Test_Hindi")




import librosa
from librosa.feature.spectral import chroma_stft

def create_chroma_array(wav_folder):
    files = listdir_ignore_hidden(wav_folder)
    chroma_arr = []
    print(len(files))
    count = 0
    pad_count = 0
    for file in files:
        #print("Converting: " + str(count))
        count = count+1
        sound = AudioSegment.from_file(os.path.join(wav_folder, file))
        samples = sound.get_array_of_samples()
        padding = False
        while(len(samples) < 22050): #Add zero padding if wav not long enough
            padding = True
            samples.append(0)
        #if padding:
            #print("ADDED PADDING")
        #print(type(samples))
        float_samples = []
        for i in range(len(samples)):
            #print(samples[i])
            float_samples.append(float(samples[i]))
        samples = np.asarray(float_samples)
        #print(type(samples))
        #print(samples)
        #print(type(samp))
        chroma = chroma_stft(samples)
        print(chroma.shape)
        chroma = np.swapaxes(chroma,0,1)
        #print(chroma.shape)
        chroma_arr.append(chroma)
        #print("WOOT")
        #print(mfcc_arr)
    print("Converted :" + str(len(files)))
    #print("TYPE IS: " + str(type(chroma_arr)))
    #mfcc_arr = np.asarray(mfcc_arr)
    #mfcc_arr = mfcc_arr.reshape(mfcc_arr.shape[0], mfcc_arr.shape[1], mfcc_arr.shape[2], 1)
    #print("TYPE IS: " + str(type(mfcc_arr)))
    return chroma_arr

#c_arr = create_chroma_array(path.dirname(__file__) + "/Audio_Data/Wav_Data/North_America")
#mfcc_arr = create_mfcc_array(path.dirname(__file__) + "/Audio_Data/Wav_Data/North_America")

#for i in range(5):
#    print(c_arr[i])
#print("(((((((((((((((((((((((((((((())))))))))))))))))))))))))))))")
#for i in range(5):
#    print(mfcc_arr[i])





#Creates a dictionary that contains an array of both the mfcc and chroma feature arrays
#Feature Dictionary looks like {'MFCC': <mfcc_feature_array>, 'Chroma': <chroma_feature_array>}
#Need to do this so that mfcc_feature_array[i] and chroma_feature_array[i] were generated -
#- from the same .wav file (Need this for cotraining)
def create_feature_dictionary(wav_folder):
    files = listdir_ignore_hidden(wav_folder)
    mfcc_arr = []
    chroma_arr = []
    print(len(files))
    count = 0
    for file in files:

        #MFCC
        #print("Converting: " + str(count))
        count = count+1
        mfcc_arr.append(create_mfcc(os.path.join(wav_folder, file)))
        #print("WOOT")
        #print(mfcc_arr)
        #print("Converting: " + str(count))


        #CHROMA
        count = count+1
        sound = AudioSegment.from_file(os.path.join(wav_folder, file))
        samples = sound.get_array_of_samples()
        padding = False
        while(len(samples) < 22050): #Add zero padding if wav not long enough
            padding = True
            samples.append(0)
        #if padding:
            #print("ADDED PADDING")
        #print(type(samples))
        float_samples = []
        for i in range(len(samples)):
            #print(samples[i])
            float_samples.append(float(samples[i]))
        samples = np.asarray(float_samples)
        #print(type(samples))
        #print(samples)
        #print(type(samp))
        chroma = chroma_stft(samples)
        #print(chroma.shape)
        chroma = np.swapaxes(chroma,0,1)
        #print(chroma.shape)
        chroma_arr.append(chroma)


    print("Converted :" + str(len(files)))
    #print("TYPE IS: " + str(type(mfcc_arr)))
    #mfcc_arr = np.asarray(mfcc_arr)
    #mfcc_arr = mfcc_arr.reshape(mfcc_arr.shape[0], mfcc_arr.shape[1], mfcc_arr.shape[2], 1)
    #print("TYPE IS: " + str(type(mfcc_arr)))
    if(len(mfcc_arr) != len(chroma_arr)):
        print("DIFFERENT LENGTHS!!!")
        print(len(mfcc_arr))
        print(len(chroma_arr))
    else:
        print("THEYRE THE SAME LENGTH!!!")
    feature_dict = {'MFCC': mfcc_arr, 'Chroma': chroma_arr}
    return feature_dict























#Removes wav files of inconsistent length from wav folder
#Finds proper length and then deletes all wavs that are not that length
def clean_wav_folder(wav_folder):
    files = listdir_ignore_hidden(wav_folder)
    print("ORIGINALLY " + str(len(files)) + " FILES")
    check_length = -1
    del_files = []
    find_len_count = 0
    cur_file = 0
    while find_len_count < 3 and cur_file < len(files):
        sound = AudioSegment.from_file(os.path.join(wav_folder, files[cur_file]))
        samples = sound.get_array_of_samples()
        if(len(samples) != check_length):
            check_length = len(samples)
            find_len_count = 0
        elif len(samples) == check_length:
            find_len_count = find_len_count + 1
        cur_file = cur_file + 1
    #check_length now holds the standard wav file length
    for file in files:
        sound = AudioSegment.from_file(os.path.join(wav_folder, file))
        samples = sound.get_array_of_samples()
        if(len(samples) != check_length):
            os.remove(os.path.join(wav_folder, file))
    new_files = listdir_ignore_hidden(wav_folder)
    print("NOW " + str(len(new_files)) + " FILES")


#NOT SURE WHY BUT THIS SEVERELY LESSENS THE CLASSIFICATION RATE
#clean_wav_folder(path.dirname(__file__) + "/Audio_Data/Wav_Data/North_America")
#clean_wav_folder(path.dirname(__file__) + "/Audio_Data/Wav_Data/India")
