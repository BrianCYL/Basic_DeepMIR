import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchaudio

class AudioData(data.Dataset):
    def __init__(self, folder_path, type):
        self.folder_path = folder_path
        if type not in ['train', 'valid', 'test']:
            raise ValueError('type should be either train, valid, or test')
        self.type = type
        self.data = []
        self.labels = []

    def load_audio(self):
        self.data = np.load(f'{self.folder_path}/{self.type}_data_mel.npy')
        self.data = torch.tensor(self.data).unsqueeze(1)
        self.labels = np.load(f'{self.folder_path}/{self.type}_label.npy')
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
class AudioDataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super(AudioDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)

class MusicNet(nn.Module):
    def __init__(self, log_scale=True):
        super(MusicNet, self).__init__()
        self.log_scale = log_scale
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.fc1 = nn.Linear(512 * 4 * 3, 512)
        self.fc2 = nn.Linear(512, 11)

    def forward(self, x):
        # x: (batch_size, 1, n_mels, time)
        # n_mels = 128, time = 126
        if self.log_scale:
            x = self.to_db(x)
        x = self.spec_bn(x)

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = x.view(-1, 512 * 4 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x