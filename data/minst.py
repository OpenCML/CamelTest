import torch
from torchvision import datasets, transforms
import numpy as np
import os

# 设置数据集存储目录
data_dir = './data/MNIST'  # 替换为你要存储数据集的路径

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor()
])

# 下载并加载MNIST数据集
train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

# 提取数据和标签，并转换为NumPy数组
x_train = train_dataset.data.numpy()
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

# 将数据保存为NumPy格式
np.save(os.path.join(data_dir, 'x_train.npy'), x_train)
np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
np.save(os.path.join(data_dir, 'x_test.npy'), x_test)
np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

print("MNIST数据集已下载并保存为NumPy格式。")
