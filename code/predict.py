import os
import torch
import torch.nn as nn
import librosa
import numpy as np



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

        # 全连接层先设为 None，等第一次前向时动态创建
        self.fc1 = None
        self.relu3 = nn.ReLU()
        self.fc2 = None

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)

        # 第一次前向时创建全连接层
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
    mfcc = mfcc.T  # (时间步, 特征维)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,时间步,特征维)
    return mfcc

# -------------------- 主程序 --------------------
if __name__ == '__main__':
    # 请在此修改为你要检测的音频文件路径
    audio_path = r"C:\Users\admin\Desktop\work\chuangxinshijian\forth\Voice_Syhz\code\noise_test.wav"

    if not os.path.exists(audio_path):
        print(f"错误：文件不存在 - {audio_path}")
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN()
    model.to(device)

    # 预处理音频
    input_tensor = preprocess_audio(audio_path).to(device)

    # 关键步骤：先用输入张量执行一次前向传播，让模型动态创建全连接层
    with torch.no_grad():
        _ = model(input_tensor)

    # 现在加载训练好的权重（此时模型已有 fc1, fc2，可以正确匹配）
    model.load_state_dict(torch.load('audio_classifier/audio_classifier2.pth', map_location=device))
    model.eval()

    # 再次前向得到预测结果
    with torch.no_grad():
        outputs = model(input_tensor)
        prob = torch.softmax(outputs, dim=1)[0, 1].item()   # AI 类的概率
        pred_class = "AI合成声" if prob > 0.5 else "真实人声"

    print(f"音频文件: {audio_path}")
    print(f"AI 合成概率: {prob:.4f} ({prob*100:.2f}%)")
    print(f"判定结果: {pred_class}")