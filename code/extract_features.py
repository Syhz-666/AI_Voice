import os
import numpy as np
import librosa


real_root = "D:\GPT-SoVITS-v3lora-20250228\output\slicer_opt\人声+AI1.1\人声_aug"
fake_root = "D:\GPT-SoVITS-v3lora-20250228\output\slicer_opt\人声+AI1.1\AI_aug"
output_file = "data/data2.npz"

def extract_mfcc(file_path, target_length=16000*7):
    """
    提取音频的MFCC特征，并统一长度
    """
    y, sr = librosa.load(file_path, sr=16000)          # 统一采样率16kHz
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))     # 短则补零
    else:
        y = y[:target_length]                          # 长则截断

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1024, hop_length=512)
    return mfcc.T   # 转置为 (时间步, 特征维)

data = []
labels = []


for root, dirs, files in os.walk(real_root):
    for file in files:
        if file.endswith('.wav'):
            path = os.path.join(root, file)
            mfcc = extract_mfcc(path)
            data.append(mfcc)
            labels.append(0)      # 真实人声标签 0


for root, dirs, files in os.walk(fake_root):
    for file in files:
        if file.endswith('.wav'):
            path = os.path.join(root, file)
            mfcc = extract_mfcc(path)
            data.append(mfcc)
            labels.append(1)      # AI合成标签 1


data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)
np.savez(output_file, data=data, labels=labels)

print(f"完成！共 {len(data)} 条样本，保存至 {output_file}")
print("特征形状:", data.shape)