from os import path
import numpy as np
import matplotlib.pyplot as plt
import os
import wave
import math
import statistics
import time
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
from pydub.playback import play

print("Hello World")

file_name = path.dirname(__file__) + "/Audio_Data/bengali6.mp3"
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

segW = AudioSegment.from_wav(path.dirname(__file__) + "/Audio_Data/Wav_Data/test.wav")

chunks = split_on_silence(segW, min_silence_len = 10, silence_thresh = -25, keep_silence = 100)

sil_ranges = detect_silence(segW)

for r in sil_ranges:
    print(str(r))

print("WE ARE HERE")
n = 0
for i, chunk in enumerate(chunks):
    n = n+1
    time.sleep(0.5)
    play(chunk)
    print(type(chunk))
print(n)

'''
from scipy.io.wavfile import read
wavdata = (read(path.dirname(__file__) + "/Audio_Data/Wav_Data/test.wav"))
print("HERE NOW")
print(type(wavdata))
print(str(wavdata))
wavdata = wavdata[1]
chunks2 = np.array_split(wavdata, 100)
print("AND ALSO HERE")
dbs = [20*math.log10( math.sqrt(statistics.mean(chunk**2)) ) for chunk in chunks2]
print("DBS:")
print(dbs)
'''
