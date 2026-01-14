import os
from typing import List

import cv2
import numpy as np
import pyrealsense2 as rs

from mediapipe_hand import MediaPipeHandDetector, HandRecords, PixelDepths

# ---------------------------------
# 从 RealSense .bag 文件中读取数据（其余逻辑和 realsense.py 一致）
# 并使用 MediaPipe 检测手部关键点，记录 21 个关键点的像素坐标、深度（米）和空间坐标（X,Y,Z）
# ---------------------------------

BAG_PATH = "./logs/raw_bags/0108_1843.bag"

def format_log_line(
    ts_ms: int,
    img_xyz: PixelDepths,
    world_xyz: HandRecords,
) -> str:
    """
    img_xyz: [(px, py, z_m), ...] length=21
    world_xyz: [(X,Y,Z), ...] length=21
    Output: timestamp_ms, px,py,z_m,X,Y,Z, ... total 1 + 21*6 columns
    """
    parts = [str(ts_ms)]
    for (px, py, z), (X, Y, Z) in zip(img_xyz, world_xyz):
        parts.extend(
            [
                str(px),
                str(py),
                f"{z:.6f}",
                f"{X:.16f}",
                f"{Y:.16f}",
                f"{Z:.16f}",
            ]
        )
    return ",".join(parts) + "\n"


def find_latest_bag(bag_dir: str) -> str:
    if not os.path.isdir(bag_dir):
        return ""
    bags = [
        os.path.join(bag_dir, name)
        for name in os.listdir(bag_dir)
        if name.lower().endswith(".bag")
    ]
    return max(bags, key=os.path.getmtime) if bags else ""


def default_log_path(bag_path: str, out_dir: str) -> str:
    base = os.path.splitext(os.path.basename(bag_path))[0]
    return os.path.join(out_dir, f"{base}_realsense_log.txt")


def main() -> None:
    bag_path = BAG_PATH or find_latest_bag("./logs/raw_bags")

    log_path = "" or default_log_path(bag_path, "./logs_bag")
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)
    hand_detector = MediaPipeHandDetector()

    buf: List[str] = []
    frame_count = 0
    max_frames = 0
    flush_every = 30

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            while True:
                try:
                    frames = pipeline.wait_for_frames()
                except RuntimeError:
                    break

                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_img = np.asanyarray(color_frame.get_data())
                intr = color_frame.profile.as_video_stream_profile().intrinsics

                img_xyz, world_xyz, all_valid = hand_detector.process(
                    color_img, depth_frame, intr
                )

                output = color_img.copy()
                if len(img_xyz) == 21:
                    for i in [0, 4, 8, 12, 16, 20]:
                        for (px, py, z) in [img_xyz[i]]:
                            color = (0, 255, 0) if z > 0 else (0, 0, 255)
                            cv2.circle(output, (px, py), 4, color, -1)
                            cv2.putText(
                                output,
                                f"{z:.2f}",
                                (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                color,
                                1,
                            )

                    ts_ms = int(color_frame.get_timestamp())
                    buf.append(format_log_line(ts_ms, img_xyz, world_xyz))

                    if len(buf) >= flush_every:
                        f.writelines(buf)
                        f.flush()
                        buf.clear()

                cv2.imshow("RealSense Bag Hands (21-point 3D)", output)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break

            if buf:
                f.writelines(buf)
                f.flush()

    finally:
        print("Done.")

        hand_detector.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
