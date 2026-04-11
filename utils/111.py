import cv2
import os

def video_to_frames(video_path, output_dir, prefix="frame", stride=1):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频！")
        return

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕

        if frame_idx % stride == 0:
            frame_name = f"{prefix}_{saved_idx:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            print(f"✅ 保存帧: {frame_path}")
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print("🎉 完成视频转帧！")

# 示例使用
video_path = "/home/admin1/datasets/DJI_test/colors/40mclolors.MP4"
output_dir = "/home/admin1/datasets/DJI_test/colors/frames"
video_to_frames(video_path, output_dir, prefix="DJI", stride=1)  # stride=1 表示每帧都保存
