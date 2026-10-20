import torch
import torch.nn as nn
import torchaudio
from torch.utils import data
import os
import json
from tqdm import tqdm
import argparse
import numpy as np
from task_3_utils import MusicNet, AudioData, AudioDataLoader

def train(model, args, train_loader, valid_loader, criterion, optimizer):
    if not os.path.exists('./checkpoint/'):
        os.makedirs('./checkpoint/')

    if torch.cuda.is_available():
        device = 'cuda'
        print('Using GPU')
    else:
        device = 'cpu'
        print('Using CPU')

    if args.log_scale:
        print("using log scale features")
        suffix = '_log'
    else:
        print("using normal scale features")
        suffix = ''

    model.to(device)
    best_loss = float('inf')
    best_epoch = 0
    for epoch in tqdm(range(args.n_epochs)):
        valid_loss = 0
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        for i, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'./checkpoint/model_best{suffix}.pth')
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'./checkpoint/model_{epoch}{suffix}.pth')
        if epoch % 10 == 0 and args.verbose:
            print('\n' + f'Epoch {epoch + 1}, Validation Loss: {valid_loss}')
            
        if epoch - best_epoch > args.early_stop:
            print('Early stopping!')
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--folder_path', type=str, default='./dataset')
    parser.add_argument('--log_scale', type=int, default=1)

    args = parser.parse_args()

    train_dataset = AudioData(args.folder_path, 'train')
    train_dataset.load_audio()
    train_loader = AudioDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = AudioData(args.folder_path, 'valid')
    valid_dataset.load_audio()
    valid_loader = AudioDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = MusicNet(args.log_scale)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(model, args, train_loader, valid_loader, criterion, optimizer) 

if __name__ == '__main__':
    main()