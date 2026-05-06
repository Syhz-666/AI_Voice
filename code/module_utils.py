import torch
import torch.nn as nn
import numpy as np
import librosa

# -------------------- 模型定义（与训练时完全一致） --------------------
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = None
        self.relu3 = nn.ReLU()
        self.fc2 = None

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)

        if self.fc1 is None:
            self.flattened_size = x.size(1)
            self.fc1 = nn.Linear(self.flattened_size, 128).to(x.device)
            self.fc2 = nn.Linear(128, 2).to(x.device)

        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------- 音频预处理函数（与训练时一致） --------------------
def preprocess_audio(file_path, target_seconds=7, sr=16000, n_mfcc=40, n_fft=1024, hop_length=512):
    y, _ = librosa.load(file_path, sr=sr)
    target_length = sr * target_seconds
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,时间步,特征维)
    return mfcc