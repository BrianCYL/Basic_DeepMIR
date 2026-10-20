from sklearn.metrics import top_k_accuracy_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
import librosa
import matplotlib.pyplot as plt
import argparse

def preprocessing(args):
    """
    perform preprocessing on the test dataset
    """
    audio_dataset = []
    folder_path = os.path.join(args.data_path, 'audio/')
    label_path = os.path.join(args.data_path, 'examples.json')
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
    return audio_dataset, label_data

def inference(args):
    # load the test data
    print('Loading data...')
    test_data, test_label = preprocessing(args)

    # normalize the features
    print('Preprocessing data...')
    scaler = StandardScaler()
    test_data = scaler.fit_transform(test_data)

    # load the model
    clf = joblib.load('./model/knn_model.pkl' , mmap_mode ='r')

    # predict the test data
    print('Predicting...')
    prob_pred = clf.predict_proba(test_data)
    pred = clf.predict(test_data)

    # calculate the performance metrics
    top_1_acc = top_k_accuracy_score(test_label, prob_pred, k=1, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f'Top 1 Accuracy: {top_1_acc}')
    top_3_acc = top_k_accuracy_score(test_label, prob_pred, k=3, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f'Top 3 Accuracy: {top_3_acc}')
    cm = confusion_matrix(test_label, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()

    if not os.path.exists('./result_image'):
        os.makedirs('./result_image')
    plt.savefig('./result_image/confusion_matrix_task_2.pdf')

def main():
    parser = argparse.ArgumentParser(description='Inference the model')
    parser.add_argument('--data_path', type=str, default='./dataset/nsynth-test', help='path to the test data')
    args = parser.parse_args()
    inference(args)


if __name__ == '__main__':
    main()
    
