"""
手势数据采集脚本
- 按数字键 0-9 切换当前要采集的手势类别
- 按空格键 保存当前帧到对应文件夹
- 按 'q' 退出
"""
import cv2
import os
from config import RAW_DATA_DIR, GESTURES

# 确保每个手势的文件夹存在
for g in GESTURES:
    folder = os.path.join(RAW_DATA_DIR, g)
    os.makedirs(folder, exist_ok=True)

# 初始手势
current_gesture = GESTURES[0]
print(f"当前手势: {current_gesture}")
print("操作说明:")
print("  按数字键 0-9 切换手势")
print("  按空格键 保存图片")
print("  按 'q' 退出")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头打开失败")
    exit()

# 统计每个类别已保存的图片数量（从文件夹中已有文件计数）
counts = {}
for g in GESTURES:
    folder = os.path.join(RAW_DATA_DIR, g)
    if os.path.exists(folder):
        counts[g] = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    else:
        counts[g] = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 镜像显示，更自然

    # 显示当前手势和已保存数量
    cv2.putText(frame, f"Gesture: {current_gesture}  Count: {counts[current_gesture]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "0-9: switch | Space: save | q: quit",
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Data Collection', frame)

    key = cv2.waitKey(1) & 0xFF

    # 按 'q' 退出
    if key == ord('q'):
        break

    # 按空格键保存
    if key == 32:  # 空格键
        folder = os.path.join(RAW_DATA_DIR, current_gesture)
        filename = f"{current_gesture}_{counts[current_gesture]:04d}.jpg"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, frame)
        counts[current_gesture] += 1
        print(f"✅ 已保存: {filepath} (总数: {counts[current_gesture]})")

    # 按数字键切换手势（键码 48-57 对应 '0'-'9'）
    if 48 <= key <= 57:
        digit = chr(key)
        if digit in GESTURES:
            current_gesture = digit
            print(f"切换到手势: {current_gesture} (已有 {counts[current_gesture]} 张)")

cap.release()
cv2.destroyAllWindows()
print("采集结束，统计如下：")
for g in GESTURES:
    print(f"  数字 {g}: {counts[g]} 张")