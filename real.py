"""
实时手势识别 - 中央裁剪 + 多帧投票 + 条形图显示
使用模型：cnn_simple.h5
"""
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from collections import Counter
from config import IMG_SIZE

# 加载模型
model = load_model('cnn_simple.h5')
print("✅ 模型加载成功")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头打开失败")
    exit()

print("按 'q' 退出")
print("💡 请将手势放在画面中央区域")

# 多帧投票参数
pred_history = []
VOTE_LENGTH = 10          # 投票队列长度
VOTE_THRESHOLD = 5        # 最少票数

# 条形图显示参数
BAR_START_X = 10          # 条状图起始X坐标
BAR_START_Y = 70          # 条状图起始Y坐标（避免挡住主结果文字）
BAR_HEIGHT = 15           # 每个条形的高度
BAR_MAX_WIDTH = 120       # 最大票数（10票）对应的条宽度
BAR_GAP = 3               # 条形之间的间距
NUM_CLASSES = 10          # 手势类别数（0-9）

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # 镜像

    # ===== 预处理：与训练完全一致 =====
    h, w = frame.shape[:2]
    top = int(h * 0.2)
    bottom = int(h * 0.8)
    left = int(w * 0.2)
    right = int(w * 0.8)
    roi = frame[top:bottom, left:right]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    input_data = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # ===== 预测 =====
    probs = model.predict(input_data, verbose=0)[0]
    pred = np.argmax(probs)
    conf = np.max(probs)

    # ===== 多帧投票（仅记录预测数字，不记录置信度） =====
    pred_history.append(pred)
    if len(pred_history) > VOTE_LENGTH:
        pred_history.pop(0)

    # ===== 统计各数字得票 =====
    vote_counts = Counter(pred_history)

    # ===== 绘制条形图 =====
    for i in range(NUM_CLASSES):
        votes = vote_counts.get(i, 0)
        # 背景条（灰色）
        bg_y = BAR_START_Y + i * (BAR_HEIGHT + BAR_GAP)
        cv2.rectangle(frame,
                      (BAR_START_X, bg_y),
                      (BAR_START_X + BAR_MAX_WIDTH, bg_y + BAR_HEIGHT),
                      (60, 60, 60), -1)
        # 票数条（绿色，宽度与票数成正比）
        bar_width = int((votes / VOTE_LENGTH) * BAR_MAX_WIDTH)
        if bar_width > 0:
            cv2.rectangle(frame,
                          (BAR_START_X, bg_y),
                          (BAR_START_X + bar_width, bg_y + BAR_HEIGHT),
                          (0, 255, 0), -1)
        # 数字标签
        cv2.putText(frame, f"{i}",
                    (BAR_START_X + BAR_MAX_WIDTH + 10, bg_y + BAR_HEIGHT - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ===== 投票决策 =====
    if len(pred_history) == VOTE_LENGTH:
        most_common = Counter(pred_history).most_common(1)[0]
        if most_common[1] >= VOTE_THRESHOLD:
            display_pred = most_common[0]
        else:
            display_pred = None
    else:
        display_pred = None

    # ===== 显示主结果 =====
    if display_pred is not None and conf > 0.3:
        label = f'{display_pred} (稳定 {conf*100:.0f}%)'
        color = (0, 255, 0)
    elif conf > 0.5:
        label = f'{pred}? ({conf*100:.0f}%)'
        color = (0, 255, 200)
    else:
        label = '?'
        color = (0, 0, 255)

    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # 画中央裁剪区域（调试用）
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)

    cv2.imshow('Real-time Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ 已退出")