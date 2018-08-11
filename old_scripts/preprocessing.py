import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import os
import glob
import numpy as np
from scipy.fftpack import dct


def list_directory(dir):
    folders = []
    for root, dirs, files in os.walk(dir):
        for name in dirs:
            folders.append(os.path.join(root, name))
    return folders

def plot_wav(samples):
    plt.figure(1)
    plt.title('Signal in Time Domain')
    plt.plot(samples)
    plt.show()

def plot_spectrogram(times, frequencies, spectrogram):
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def mfcc(spectrogram):
    mfcc = dct(spectrogram,type = 2, axis = 1, norm='ortho')
    return mfcc

def parse_pandas(folders,adjust_volume):
    '''
    :param folders: list of folder path to be parsed
    :param adjust_volume: if the volume should be adjusted by peak(default)/RMS 
    :return: 
    '''
    parsed_dict = {}
    for folder in folders:
        counter = 0
        for file in glob.glob(os.path.join(os.path.join(data_dir, folder), '*.wav')):
            sample_rate, samples = wavfile.read(os.path.join(folder, file))
            # pre-emphasis filter to (1) balance between high/low frequency (2) avoid fourier numerical issues
            np.append(samples[0],samples[1:]-0.97*samples[:-1])
            # normalization
            if adjust_volume:
                samples = samples/max(samples)
            # spectrogram
            frequencies, times, spectrogram = signal.spectrogram(samples,sample_rate)
            # mfcc
            spectrogram = mfcc(spectrogram)
            parsed_dict[folder.split('/')[-1]+str(counter)] = [frequencies, times, spectrogram,folder.split('/')[-1]]
            counter += 1
    return parsed_dict

data_dir = '/Users/weiyuchen/Documents/Project/VoiceCommand/Dataset'
folders = list_directory(data_dir)
parsed_dict = parse_pandas([folders[-2],folders[-8]],adjust_volume = True)

with open('training_v3.pickle', 'wb') as handle:
    pickle.dump(parsed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

