from os import path, listdir
import numpy as np
import matplotlib.pyplot as plt
import os
import wave
import math
import statistics
import time
import re
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
from pydub.playback import play


#List all files in a directory except hidden files
def listdir_ignore_hidden(path):
    files = []
    for file in os.listdir(path):
        if not file.startswith('.'):
            files.append(file)
    return files


print("Hello World")

file_name = path.dirname(__file__) + "/Audio_Data/Mp3_Data/bengali6.mp3"
output_file_name = path.dirname(__file__) + "/Audio_Data/Wav_Data/bengali6.wav"


print("HELLO")
print(os.environ['PATH'])
print("GOODBYE")

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


from scipy.io.wavfile import read
wavdata = (read(path.dirname(__file__) + "/Audio_Data/Wav_Data/bengali6.wav"))
print("HERE NOW")
print(type(wavdata))
print(str(wavdata))
wavdata = wavdata[1]
chunks2 = np.array_split(wavdata, 1)
print("AND ALSO HERE")
dbs = [20*math.log10( math.sqrt(statistics.mean(chunk**2)) ) for chunk in chunks2]
print("DBS:")
print(dbs)

# [/\\]{1}([^/\\]+\w+)[\.]   <- REGEX FOR PARSING NAME OF FILE FROM FILE PATH


def extract_chunks(mp3File):
    wavFile = convert_to_wav(mp3File)
    avgDb = calculate_avg_db(wavFile)

#INPUT: 1) Fully qualified name of mp3 file, 2) Fully qualified name of folder in which wav file will be created
#OUTPUT: A wav file with the same name as the mp3 file will be created in the specified output_folder
#i.e. hello.mp3 -> hello.wav
def convert_to_wav(mp3File, output_folder):
    file_name = os.path.basename(mp3File)
    file_name = file_name.split('.')[0]
    AudioSegment.from_mp3(mp3File).export(output_folder + file_name +".wav", format="wav")
    return

#INPUT: 1) Fully qualified name of wav file
#OUTPUT: Average volume in dB of file
def calculate_avg_db(wavFile):
    wavdata = (read(wavFile))
    wavdata = wavdata[1]
    chunks = np.array_split(wavdata, 1)
    dbs = [20*math.log10( math.sqrt(statistics.mean(chunk**2)) ) for chunk in chunks]
    print(dbs)
    return dbs

convert_to_wav(path.dirname(__file__) + "/Audio_Data/Mp3_Data/LongIslandGirls.mp3", path.dirname(__file__) + "/Audio_Data/Wav_Data/")
