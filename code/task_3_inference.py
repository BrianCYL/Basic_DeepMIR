import torch
import torch.nn as nn
import torchaudio
import os
import json
from tqdm import tqdm
import argparse
import numpy as np
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, top_k_accuracy_score
import matplotlib.pyplot as plt
from task_3_utils import MusicNet

def inference(args):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if args.log_scale:
        suffix = '_log'
    else:
        suffix = ''

    model = MusicNet(args.log_scale)
    model.load_state_dict(torch.load(f'./model/model_best{suffix}.pth'))
    model.to(device)
    model.eval()

    folder_path = os.path.join(args.data_path, 'audio/')
    label_path = os.path.join(args.data_path, 'examples.json')
    test_data_files = os.listdir(folder_path)
    test_labels = []
    with open(label_path, 'r') as f:
        label_file = json.load(f)
    preds = []
    prob_preds = []
    correct = 0
    for file in tqdm(test_data_files):
        if file[-4:] != '.wav':
            continue
        file_path = folder_path + file
        data, sr = librosa.load(file_path, sr=None) 
        data = librosa.feature.melspectrogram(y=data, sr=sr)
        data = torch.tensor(data).unsqueeze(0)
        data = data.to(device)
        data = data.unsqueeze(0)

        prob_pred = model(data)
        prob_pred = nn.Softmax(dim=1)(prob_pred)
        pred = np.argmax(prob_pred.cpu().detach().numpy(), axis=1)
        prob_pred = prob_pred.cpu().detach().numpy()
        preds.append(pred)
        prob_preds.append(prob_pred)

        label = label_file[file[:-4]]['instrument_family']
        test_labels.append(label)

    preds = np.concatenate(preds, axis=0)
    prob_preds = np.concatenate(prob_preds, axis=0)

    return preds, prob_preds, test_labels

def accuracy_report(args, preds, prob_preds, test_labels):
    top_1_acc = top_k_accuracy_score(test_labels, prob_preds, k=1, labels=[i for i in range(11)])
    print(f'Top 1 Accuracy: {top_1_acc}')
    top_3_acc = top_k_accuracy_score(test_labels, prob_preds, k=3, labels=[i for i in range(11)])
    print(f'Top 3 Accuracy: {top_3_acc}')
    cm = confusion_matrix(test_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(11)])
    disp.plot()

    if args.log_scale:
        suffix = '_log'
    else:
        suffix = ''

    if not os.path.exists('./result_image'):
        os.makedirs('./result_image')
    plt.savefig(f'./result_image/confusion_matrix_task_3{suffix}.pdf')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_scale', type=int, default=True)

    args = parser.parse_args()

    preds, prob_preds, test_labels = inference(args)
    accuracy_report(args, preds, prob_preds, test_labels)


if __name__ == '__main__':
    main()