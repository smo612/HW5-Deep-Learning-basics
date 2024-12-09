import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import datetime
import os
import shutil

# 新增：刪除舊的 TensorBoard 日誌資料
log_dir_base = "logs/fit"
if os.path.exists(log_dir_base):
    shutil.rmtree(log_dir_base)  # 刪除舊的日誌目錄
os.makedirs(log_dir_base)  # 重新建立日誌目錄

# 載入 Iris 資料集
iris = load_iris()
X = iris.data
y = iris.target

# 將資料集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 資料標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 將標籤轉為 one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

# 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 設定 TensorBoard 回呼函數
log_dir = os.path.join(log_dir_base, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 訓練模型
model.fit(X_train, y_train,
          epochs=50,
          validation_data=(X_test, y_test),
          callbacks=[tensorboard_callback])

# 評估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# 啟動 TensorBoard
# 在命令行中輸入以下指令以啟動 TensorBoard
# tensorboard --logdir=logs/fit