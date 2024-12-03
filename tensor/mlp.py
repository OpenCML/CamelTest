import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置为2以忽略INFO和WARNING

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 加载NumPy数据
x_train = np.load('data/MNIST/x_train.npy')
y_train = np.load('data/MNIST/y_train.npy')
x_test = np.load('data/MNIST/x_test.npy')
y_test = np.load('data/MNIST/y_test.npy')

# 展平输入数据（将28x28的图像展平成784的一维数组）
x_train = x_train.reshape(-1, 28 * 28).astype('float32')
x_test = x_test.reshape(-1, 28 * 28).astype('float32')

# 将标签转换为one-hot编码
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 构建MLP模型
model = models.Sequential()
model.add(layers.Input(shape=(28 * 28,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc}')
