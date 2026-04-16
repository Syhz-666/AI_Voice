import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示（和你现有可视化脚本一致）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 从训练输出中提取的关键数据（按Epoch 5/10/15/20/25/30整理）
epochs = [5, 10, 15, 20, 25, 30]
train_loss = [0.2410, 0.1168, 0.0646, 0.0332, 0.0180, 0.0120]
val_loss = [0.2664, 0.1801, 0.1576, 0.1636, 0.1602, 0.1716]
train_acc = [0.9343, 0.9710, 0.9886, 0.9981, 1.0000, 1.0000]
val_acc = [0.9091, 0.9293, 0.9470, 0.9419, 0.9419, 0.9343]

# 创建2行1列的子图（损失+准确率）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 绘制损失曲线
ax1.plot(epochs, train_loss, 'o-', label='训练损失', color='#1f77b4', linewidth=2)
ax1.plot(epochs, val_loss, 's-', label='验证损失', color='#ff7f0e', linewidth=2)
ax1.set_xlabel('训练轮数（Epoch）')
ax1.set_ylabel('损失值（CrossEntropyLoss）')
ax1.set_title('训练/验证损失变化曲线')
ax1.legend()
ax1.grid(alpha=0.3)

# 绘制准确率曲线
ax2.plot(epochs, train_acc, 'o-', label='训练准确率', color='#1f77b4', linewidth=2)
ax2.plot(epochs, val_acc, 's-', label='验证准确率', color='#ff7f0e', linewidth=2)
ax2.set_xlabel('训练轮数（Epoch）')
ax2.set_ylabel('准确率（Accuracy）')
ax2.set_title('训练/验证准确率变化曲线')
ax2.set_ylim(0.6, 1.05)  # 限定纵坐标范围，突出差异
ax2.legend()
ax2.grid(alpha=0.3)

# 调整布局并保存
plt.tight_layout()
plt.savefig('train_results/train_results2.png', dpi=150, bbox_inches='tight')
plt.show()

print("训练结果曲线已保存为 train_results.png")