import os
import json
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
def extract_mfcc(file_path, target_seconds=7, sr=16000, n_mfcc=40, n_fft=1024, hop_length=512):
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


# -------------------- 批量预测函数 --------------------
def batch_predict(audio_paths, model, device):

    results = []
    model.eval()  # 设为评估模式

    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            print(f"警告：文件不存在 - {audio_path}，跳过")
            continue

        # 预处理音频
        input_tensor = extract_mfcc(audio_path).to(device)

        # 预测
        with torch.no_grad():
            outputs = model(input_tensor)
            prob = torch.softmax(outputs, dim=1)[0, 1].item()  # AI 类的概率
            pred_class = "AI合成声" if prob > 0.5 else "真实人声"

        # 提取文件名
        file_name = os.path.basename(audio_path)

        # 构造单条结果
        result = {
            "文件名": file_name,
            "AI合成概率": round(prob, 4),
            "AI合成概率(百分比)": f"{prob * 100:.2f}%",
            "判定结果": pred_class
        }

        # 打印控制台（保留原格式+文件名）
        print(f"音频文件: {file_name}")
        print(f"AI 合成概率: {prob:.4f} ({prob * 100:.2f}%)")
        print(f"判定结果: {pred_class}")
        print("-" * 50)

        results.append(result)
    return results


# -------------------- 主程序 --------------------
if __name__ == '__main__':

    ai_audio_root = r"D:\GPT-SoVITS-v3lora-20250228\output\slicer_opt\AI\3-1"  # AI音频文件夹
    real_audio_root = r"D:\GPT-SoVITS-v3lora-20250228\output\slicer_opt\人声\3"  # 人声音频文件夹
    model_path = "audio_classifier/audio_classifier.pth"  # 训练好的模型路径
    output_json_path = "predict_results/predict_results.json"  # 结果保存的JSON文件路径


    audio_paths = []    # 收集所有验证音频路径

    data = []
    labels = []

    for root, dirs, files in os.walk(real_audio_root):
        for file in files:
            if file.endswith('.wav'):
                path = os.path.join(root, file)
                mfcc = extract_mfcc(path)
                data.append(mfcc)
                labels.append(0)  # 真实人声标签 0
                audio_paths.append(os.path.join(root, file))

    for root, dirs, files in os.walk(ai_audio_root):
        for file in files:
            if file.endswith('.wav'):
                path = os.path.join(root, file)
                mfcc = extract_mfcc(path)
                data.append(mfcc)
                labels.append(1)  # AI合成标签 1
                audio_paths.append(os.path.join(root, file))

    if not audio_paths:
        print("错误：未找到任何.wav格式的音频文件")
        exit(1)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = AudioCNN()
    model.to(device)

    # 先执行一次前向传播创建全连接层
    dummy_input = extract_mfcc(audio_paths[0]).to(device)
    with torch.no_grad():
        _ = model(dummy_input)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("模型加载完成！")
    print(f"共找到 {len(audio_paths)} 个验证音频文件")
    print("开始预测...")
    print("-" * 50)

    # 批量预测
    results = batch_predict(audio_paths, model, device)

    # 保存结果到JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n预测完成！结果已保存至 {output_json_path}")