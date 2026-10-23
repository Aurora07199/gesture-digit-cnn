"""
模型评估：混淆矩阵
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import os
from config import RAW_DATA_DIR, GESTURES, IMG_SIZE
from sklearn.model_selection import train_test_split
import cv2

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("加载测试数据...")
print("=" * 60)

X, y = [], []
for label in GESTURES:
    folder = os.path.join(RAW_DATA_DIR, label)
    if not os.path.exists(folder):
        continue
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
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

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = load_model('cnn_gesture.h5')
print("✅ 模型加载成功")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=GESTURES))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=GESTURES, yticklabels=GESTURES)
plt.title('CNN 混淆矩阵', fontsize=15)
plt.xlabel('预测标签', fontsize=13)
plt.ylabel('真实标签', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
print("✅ 混淆矩阵已保存")