import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# -------------------- 1. 加载数据 --------------------
data_dict = np.load("data/data2.npz")
X = data_dict["data"]          # (样本数, 时间步, 特征维)
y = data_dict["labels"]        # (样本数,)

print("原始数据形状:", X.shape)
print("标签形状:", y.shape)

# -------------------- 2. 划分训练/验证集 --------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 转换为 PyTorch 张量，并添加通道维度 (batch, channel, height, width)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)   # (N, 1, 时间步, 特征维)
X_val   = torch.tensor(X_val,   dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val,   dtype=torch.long)

print("训练集形状:", X_train.shape)   # (N_train, 1, height, width)
print("验证集形状:", X_val.shape)

# -------------------- 3. 定义模型（自动适配输入尺寸） --------------------
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # 全连接层先设为 None，等第一次前向时根据实际尺寸创建
        self.fc1 = None
        self.relu3 = nn.ReLU()
        self.fc2 = None

    def forward(self, x):
        # 卷积部分
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # 展平
        x = x.view(x.size(0), -1)

        # 动态创建全连接层
        if self.fc1 is None:
            self.flattened_size = x.size(1)
            self.fc1 = nn.Linear(self.flattened_size, 128).to(x.device)
            self.fc2 = nn.Linear(128, 2).to(x.device)

        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = AudioCNN()
print(model)

# -------------------- 4. 训练设置 --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

epochs = 30

# -------------------- 5. 训练循环 --------------------
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()

    train_acc = train_correct / len(train_dataset)
    train_loss = train_loss / len(train_dataset)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / len(val_dataset)
    val_loss = val_loss / len(val_dataset)

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | 训练损失: {train_loss:.4f} 准确率: {train_acc:.4f} | 验证损失: {val_loss:.4f} 准确率: {val_acc:.4f}")

# -------------------- 6. 保存模型 --------------------
torch.save(model.state_dict(), "audio_classifier/audio_classifier2.pth")
print("模型已保存为 audio_classifier.pth")