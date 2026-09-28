# MNIST 手写数字识别（PyTorch）

用 PyTorch 搭了一个全连接神经网络来识别 MNIST 手写数字。
从跑通代码到调参，最后测试准确率到了 98.09%。

A fully-connected neural network built with PyTorch for MNIST handwritten digit classification.
Final test accuracy: 98.09%.

---

## 环境 / Environment

- Python 3.14
- PyTorch 2.14.0
- NumPy
- 数据集：`mnist.npz`，放在 `data/` 目录下

```bash
pip install torch numpy
```
---

## 网络结构 / Model

```text
Flatten(28×28 → 784)
Linear(784, 256) + ReLU
Linear(256, 128) + ReLU
Linear(128, 64) + ReLU
Linear(64, 10)
```
3 个隐藏层，逐步从 256 缩小到 64，最后输出 10 个类别。

Three hidden layers, shrinking from 256 down to 64, then outputting 10 classes.

---

## 调参记录 / Experiments

| 实验 | 网络结构 | 学习率 | Epoch | 准确率 |
|------|----------|--------|-------|--------|
| 1 | 784→128→10 | 0.001 | 5 | 97.17% |
| 2 | 784→256→10 | 0.001 | 5 | 97.70% |
| 3 | 784→512→10 | 0.001 | 5 | 97.85% |
| 4 | 784→5120→10 | 0.001 | 5 | 97.56% |
| 5 | 784→256→128→64→10 | 0.001 | 10 | **98.09%** |
| 6 | 784→512→256→128→64→10 | 0.001 | 10 | 98.06% |

---

## 几点观察 / Notes

中间层从 128 加到 512，准确率有小幅提升；加到 5120 反而降了，而且训练明显变慢。参数太多容易过拟合。

Adding width from 128 to 512 gave a small boost. Going up to 5120 actually hurt accuracy and slowed training down significantly — too many parameters lead to overfitting.

加深网络（多堆几层）比单纯加宽效果更好，参数也更少。

Making the network deeper worked better than making it wider, and used fewer parameters.

但深度也不是越深越好。实验 6 比实验 5 多了一层（512→256→128→64），准确率反而略降（98.06% vs 98.09%）。这个全连接网络在 MNIST 上大概到 98% 左右就到顶了。

But deeper isn't always better. Experiment 6 had one more layer than Experiment 5 (512→256→128→64), yet accuracy dropped slightly (98.06% vs 98.09%). This fully-connected network seems to plateau around 98% on MNIST.

学习率 0.01 太大，准确率掉到 97% 左右；0.0001 太小，10 个 epoch 还没收敛。0.001 对这个网络刚好。

A learning rate of 0.01 was too large (accuracy dropped to around 97%). 0.0001 was too small and hadn't converged after 10 epochs. 0.001 was the sweet spot.

网络变深之后，epoch 也要跟着加。5 个 epoch 不够，10 个 epoch 才基本收敛。

Deeper networks need more epochs. 5 wasn't enough; 10 was roughly where it converged.

数据集直接加载本地的 mnist.npz，没有用 torchvision.datasets 在线下载，避免网络问题。

The dataset is loaded from a local mnist.npz instead of downloading via torchvision.datasets, to avoid network issues.

---

## 运行 / Run

```bash
python MNIST.py
```

---

## 说明 / Note

代码参考了 PyTorch 官方教程和公开学习资料，在此基础上完成了环境配置、数据加载适配、超参数调整和结果记录。

The code is based on the PyTorch official tutorial and other public learning materials. Environment setup, data loading adaptation, hyperparameter tuning, and result logging were done on top of that.