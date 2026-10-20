import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import json

def load_audio_with_librosa(filename, type='melspec'):
    """
    load the wav file using librosa
    :param filename: the path to the audio file
    :param type: the type of feature extraction (melspectrogram or mfcc)
    :return: the audio data
    """
    audio, sr = librosa.load(filename, sr=None) 
    if type == 'melspec':
        result = librosa.feature.melspectrogram(y=audio, sr=sr)
    elif type == 'mfcc':
        result = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        result = np.mean(result, axis=1)
    else:
        raise ValueError('Invalid feature type')
    return result, sr

def main():
    for filetype in ['train', 'valid']:
        mel_audio_dataset = []
        mfcc_audio_dataset = []
        folder_path = f'./dataset/nsynth-{filetype}/audio/'
        label_path = f'./dataset/nsynth-{filetype}/examples.json'
        list_dir = os.listdir(folder_path)
        labels = []
        with open(label_path, 'r') as f:
            label_file = json.load(f)
        for file in tqdm(list_dir):
            mel_audio_data, sr = load_audio_with_librosa(folder_path + file, type='melspec')
            mfcc_audio_data, sr = load_audio_with_librosa(folder_path + file, type='mfcc')
            mel_audio_dataset.append(mel_audio_data)
            mfcc_audio_dataset.append(mfcc_audio_data)
            labels.append(label_file[file[:-4]]['instrument_family'])

        mel_audio_dataset = np.array(mel_audio_dataset)
        mfcc_audio_dataset = np.array(mfcc_audio_dataset)
        label_data = np.array(labels)
        np.save(f'./dataset/{filetype}_data_mel.npy', mel_audio_dataset)
        np.save(f'./dataset/{filetype}_data_mfcc.npy', mfcc_audio_dataset)
        np.save(f'./dataset/{filetype}_label.npy', label_data)


if __name__ == '__main__':
    main()