import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 数据：图片变数字向量，像素缩到 0~1
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size=1000)

# 2. 网络结构
model = nn.Sequential(
    nn.Flatten(),                    # 28×28 图片拉平成 784 个数字
    nn.Linear(784, 512),       # 加权：784 → 512
    nn.ReLU(),                       # 多次激活：弯折  提高整体准确率
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),       # 输出：64 → 10（对应 0~9）
)

# 3. 损失与优化器
criterion = nn.CrossEntropyLoss()   # 包含 softmax
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
for epoch in range(10):              # 进行十轮训练
    for images, labels in train_loader:
        optimizer.zero_grad()       # 清空旧梯度
        output = model(images)      # 前向传播
        loss = criterion(output, labels)  # 误差分析
        loss.backward()             # 反向传播
        optimizer.step()            # 更新参数
    print(f"Epoch {epoch+1} 训练完成")

# 5. 测试准确率
correct = 0
for images, labels in test_loader:
    output = model(images)
    pred = output.argmax(dim=1)     # 取概率极大值
    correct += (pred == labels).sum().item()
print(f"测试准确率: {correct / 10000 * 100:.2f}%")