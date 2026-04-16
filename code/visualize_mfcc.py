import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题
import os

# 加载数据
data_dict = np.load("data.npz")
X = data_dict["data"]          # 形状 (样本数, 时间步, 特征维)
y = data_dict["labels"]        # 标签 0=真实, 1=AI


real_indices = np.where(y == 0)[0]
ai_indices = np.where(y == 1)[0]

# 随机各选3个样本
n_samples = min(3, len(real_indices), len(ai_indices))
real_selected = np.random.choice(real_indices, n_samples, replace=False)
ai_selected = np.random.choice(ai_indices, n_samples, replace=False)


fig, axes = plt.subplots(2, n_samples, figsize=(5*n_samples, 8))
fig.suptitle("MFCC 特征对比：真实人声 vs AI合成声", fontsize=16)


for i, idx in enumerate(real_selected):
    mfcc = X[idx]  # (时间步, 特征维)
    ax = axes[0, i]
    im = ax.imshow(mfcc.T, aspect='auto', origin='lower',
                   cmap='viridis', interpolation='nearest')
    ax.set_title(f"真实人声 {i+1}")
    ax.set_xlabel("时间帧")
    ax.set_ylabel("MFCC 系数 (1~40)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


for i, idx in enumerate(ai_selected):
    mfcc = X[idx]
    ax = axes[1, i]
    im = ax.imshow(mfcc.T, aspect='auto', origin='lower',
                   cmap='viridis', interpolation='nearest')
    ax.set_title(f"AI合成声 {i+1}")
    ax.set_xlabel("时间帧")
    ax.set_ylabel("MFCC 系数 (1~40)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("mfcc_comparison.png", dpi=150)
plt.show()

print("对比图已保存为 mfcc_comparison.png")