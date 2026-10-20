import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import json
import librosa
from tqdm import tqdm

def preprocessing():
    """
    perform preprocessing on the test dataset
    """
    audio_dataset = []
    folder_path = f'./dataset/nsynth-test/audio/'
    label_path = f'./dataset/nsynth-test/examples.json'
    list_dir = os.listdir(folder_path)
    labels = []
    with open(label_path, 'r') as f:
        label_file = json.load(f)
    for file in list_dir:
        audio, sr = librosa.load(folder_path + file, sr=None)
        audio_data = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        audio_data = np.mean(audio_data, axis=1)
        audio_dataset.append(audio_data)
        labels.append(label_file[file[:-4]]['instrument_family'])
    audio_dataset = np.array(audio_dataset)
    label_data = np.array(labels)
    np.save(f'./dataset/test_data_mfcc.npy', audio_dataset)
    np.save(f'./dataset/test_label.npy', label_data)
    return audio_dataset, label_data

def fit_model(train_data, train_label, valid_data, valid_label, search_range=[1, 10, 20, 30, 40, 50]):
    """
    fit the kNN model using the training data and tuning with the validation data
    :param train_data: the training data
    :param train_label: the training label
    :param valid_data: the validation data
    :param valid_label: the validation label
    """
    best_params = {}
    best_score = 0
    param_set = {'k': search_range}

    if not os.path.exists('./model'):
        os.makedirs('./model')

    # search for the best hyperparameter
    for k in tqdm(param_set['k']):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train_data, train_label)
        pred = neigh.predict(valid_data)
        acc = accuracy_score(valid_label, pred)
        if acc > best_score:
            best_params['acc'] = acc
            best_params['k'] = k
            joblib.dump(neigh, './model/knn_model.pkl')

    return best_params

def main():
    # load the training and validation data
    print('Loading data...')
    train_data = np.load('./dataset/train_data_mfcc.npy')
    train_label = np.load('./dataset/train_label.npy')
    valid_data = np.load('./dataset/valid_data_mfcc.npy')
    valid_label = np.load('./dataset/valid_label.npy')

    # normalize the features
    print('Preprocessing data...')
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    valid_data = scaler.fit_transform(valid_data)

    # train the model
    print('Training model...')
    search_range = [5*i for i in range(1, 11)]
    params = fit_model(train_data, train_label, valid_data, valid_label, search_range)

    print('DONE!\n')
    print(f'Best parameters: {params}')


    

if __name__ == '__main__':
    main()

