"""
配置文件
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径（改成你的实际路径）
RAW_DATA_DIR = r"D:\Sign-Language-Digits-Dataset-master\Sign-Language-Digits-Dataset-master\Dataset"

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

GESTURES = [str(i) for i in range(10)]
NUM_CLASSES = 10
IMG_SIZE = 64
CHANNELS = 1

TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 5

for d in [MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"✅ 数据路径: {RAW_DATA_DIR}")
print(f"✅ 路径存在: {os.path.exists(RAW_DATA_DIR)}")