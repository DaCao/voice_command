import numpy as np
import os
import re
import pickle
import json
import tarfile,sys
from scipy import signal
from scipy.io import wavfile
from sklearn.preprocessing import normalize
import glob
from scipy.fftpack import dct
import shutil
import random




class DataDude():
    def __init__(self):
        self.lexicon_txt_path = 'data/aishell/resource_aishell/lexicon.txt'
        self.wav_path = 'data/aishell/data_aishell/wav'
        self.dev = os.path.join(self.wav_path, 'dev')
        self.test = os.path.join(self.wav_path, 'test')
        self.train = os.path.join(self.wav_path, 'train')

        self.d_char_pho, self.d_pho_char = self.load_lexicon_from_json()
        self.d_id_script = self.load_transcripts_dict()



    def save_lexicon_to_json(self):
        path = 'data/aishell/resource_aishell/lexicon.txt'
        d_char_pho = {}
        d_pho_char = {}
        count = 0
        with open(path, 'r') as lexicon_f:
            for line in lexicon_f.readlines():
                line = line.strip()
                chn_word, phoneme = line.split(' ')[0], ' '.join(line.split(' ')[1:])
                d_char_pho[chn_word] = phoneme
                d_pho_char[phoneme] = chn_word
                count += 1

        with open('data/aishell/d_char_pho.json', 'w') as file:
            json.dump(d_char_pho, file)
        with open('data/aishell/d_pho_char.json', 'w') as file:
            json.dump(d_pho_char, file)

        print('%s items saved in dictionary'%count)
        return 0


    def load_lexicon_from_json(self):
        with open('data/aishell/d_char_pho.json', 'r') as file:
            d_char_pho = json.load(file)
        with open('data/aishell/d_pho_char.json', 'r') as file:
            d_pho_char = json.load(file)
        return d_char_pho, d_pho_char


    def save_transcript_to_json(self):
        path = 'data/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt'
        d_id_script = {}
        count = 0
        with open(path, 'r') as lexicon_f:
            for line in lexicon_f.readlines():
                line = line.strip()
                id, script = line.split(' ')[0], ' '.join(line.split(' ')[1:])
                d_id_script[id] = script
                count += 1

        with open('data/aishell/d_id_script.json', 'w') as file:
            json.dump(d_id_script, file)

        print('%s scripts saved in dictionary'%count)
        return 0


    def load_transcripts_dict(self):
        with open('data/aishell/d_id_script.json', 'r') as file:
            d_id_script = json.load(file)
        return d_id_script


    def unzip_file(self):
        for filename in os.listdir(self.wav_path):
            if filename.endswith(".tar.gz"):
                file_path = os.path.join(self.wav_path, filename)
                try:
                    tar = tarfile.open(file_path, "r:gz")
                    tar.extractall(path=self.wav_path)
                    tar.close()
                    print("Extracted %s in Directory: %s"%(file_path, self.wav_path))
                except:
                    print("unzip fail: %s" %file_path)


    def wav_to_feature(self, wav_file_path):
        try:
            sample_rate, samples = wavfile.read(wav_file_path)
            # pre-emphasis filter to (1) balance between high/low frequency (2) avoid fourier numerical issues
            np.append(samples[0],samples[1:]-0.97*samples[:-1])
            # normalization
            samples = samples/max(samples)
            # spectrogram
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
            # mfcc
            spectrogram = self.mfcc(spectrogram)  # num_features by num_time_steps
        except:
            print('wav_to_feature failed on %s'%wav_file_path)
        return spectrogram


    def mfcc(self, spectrogram):
        mfcc = dct(spectrogram,type = 2, axis = 1, norm='ortho')
        return mfcc

    def load_data(self):
        with open('filename.pickle', 'rb') as handle:
            b = pickle.load(handle)


    def move_files(self, train_test_dev):
        dest_path = os.path.join(self.wav_path, train_test_dev)
        for folder in os.listdir(dest_path):
            if os.path.isdir(os.path.join(dest_path, folder)):
                for file in os.listdir(os.path.join(dest_path, folder)):
                    try:
                        shutil.move(os.path.join(dest_path, folder, file), dest_path)
                    except:
                        continue


    def load_batch_data(self, train_test_dev, batch_size):
        # d_char_pho, d_pho_char = self.load_lexicon_from_json()
        # d_id_script = self.load_transcripts_dict()

        work_folder = os.path.join(self.wav_path, train_test_dev)
        wav_file_names = os.listdir(work_folder)
        chosen_indexes = random.sample(range(len(wav_file_names)), batch_size)
        print(chosen_indexes)

        d = {}
        for ind in chosen_indexes:
                try:
                    full_path = os.path.join(work_folder, wav_file_names[ind])
                    feature = self.wav_to_feature(full_path)
                    id = wav_file_names[ind].split('.')[0]
                    label = ' '.join([self.d_char_pho[word] for word in self.d_id_script[id].split()])
                    d[id] = {'feature': feature, 'label': label}
                    #print(full_path, label, feature.shape)
                except:
                    print(full_path, ' failed ')
                    continue
        return d


    def preprocess_features_and_labels(self, train_test_dev):
        if train_test_dev == 'train':
            f = self.train
        elif train_test_dev == 'test':
            f = self.test
        elif train_test_dev == 'dev':
            f = self.dev

        d_char_pho, d_pho_char = self.load_lexicon_from_json()
        d_id_script = self.load_transcripts_dict()

        d = {}

        for folder in os.listdir(f):
            for wav_file in os.listdir(os.path.join(f, folder)):
                if wav_file.endswith('.wav'):
                    try:
                        full_path = os.path.join(f, folder, wav_file)
                        # print(full_path)
                        feature = self.wav_to_feature(full_path)
                        id = wav_file.split('.')[0]
                        label = ' '.join([d_char_pho[word] for word in d_id_script[id].split()])
                        d[id] = {'feature': feature, 'label': label}
                    except:
                        print(full_path, ' failed ')
                        continue

        return d




# dude.preprocess_features_and_labels('train')
# dude.preprocess_features_and_labels('test')
# dude.preprocess_features_and_labels('dev')

# dude.move_files('test')
# dude.move_files('train')
# dude.move_files('dev')

dude = DataDude()

dude.load_batch_data('test', 20)