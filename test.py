import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
def get_data(num_points=1000):
    x = np.linspace(-2 * np.pi, 2 * np.pi, num_points)
    y = np.cos(x)
    return x, y

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# 数据准备
x_np, y_np = get_data(1000)
x_train = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

# 模型、损失函数和优化器
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练
epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# 预测
model.eval()
with torch.no_grad():
    y_pred = model(x_train).squeeze().numpy()

# 画图
plt.figure(figsize=(10, 5))
plt.plot(x_np, y_np, label='真实cos(x)')
plt.plot(x_np, y_pred, label='MLP拟合', linestyle='--')
plt.legend()
plt.title('MLP拟合cos(x)函数')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
import os

# 创建model文件夹（如果不存在）
if not os.path.exists('model'):
    os.makedirs('model')

# 保存模型参数
torch.save(model.state_dict(), 'model/mlp.pth')
print("模型参数已保存到 model/mlp.pth")
