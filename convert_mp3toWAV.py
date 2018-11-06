from os import path
from pydub import AudioSegment
def convert_mp3toWAV(filename):
    src=filename
    dst="test1.wav"
    sound=AudioSegment.from_mp3(src)
    sound.export(dst,format="wav")