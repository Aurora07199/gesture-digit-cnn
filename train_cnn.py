"""
最简训练配置 - 先保证模型能学到东西
"""
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from config import RAW_DATA_DIR, GESTURES, IMG_SIZE

print("="*60)
print("加载数据...")
print("="*60)

X, y = [], []
for label in GESTURES:
    folder = os.path.join(RAW_DATA_DIR, label)
    if not os.path.exists(folder):
        continue
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    print(f"数字 {label}: {len(files)} 张")
    for fname in files:
        img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        X.append(img)
        y.append(int(label))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
print(f"\n✅ 总共加载: {len(X)} 张图片")

# 划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"训练集: {len(X_train)}，验证集: {len(X_val)}，测试集: {len(X_test)}")

# ----- 构建模型（简单CNN，无L2正则化）-----
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----- 训练（无数据增强）-----
print("\n开始训练...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)],
    verbose=1
)

# ----- 测试 -----
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n测试准确率: {acc*100:.2f}%")

model.save('cnn_simple.h5')
print("✅ 模型已保存: cnn_simple.h5")