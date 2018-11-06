import librosa
import librosa.display
import os.path
import matplotlib.pyplot as plt
def convert_WAVtoMFCC(filename,destination_folder):
    y,sr=librosa.load(filename+".wav")
    print("*******************************************************************")
    mfccs=librosa.feature.mfcc(y=y, sr=sr,n_mfcc=40)
    print(mfccs)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    graph_name="{}{}".format(filename,".png")
    plt.savefig(os.path.join(destination_folder,graph_name))
    plt.show()
convert_WAVtoMFCC("test1","X:\\CS 688\\Project\\Accent Data\\India\\Hindi\\MFCC plot")
#filename,destination folder