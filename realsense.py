import os
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from mediapipe_hand import MediaPipeHandDetector, HandRecords, PixelDepths


def format_log_line(
    ts_ms: int,
    img_xyz: PixelDepths,
    world_xyz: HandRecords,
) -> str:
    """
    img_xyz: [(px, py, z_m), ...] length=21
    world_xyz: [(X,Y,Z), ...] length=21
    输出：timestamp_ms, px,py,z_m,X,Y,Z, ... 共 1 + 21*6 列
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


def main():
    rgb_dir = f"./logs/rgb_videos"
    bag_dir = f"./logs/raw_bags"

    date = datetime.now().strftime("%m%d")
    time = datetime.now().strftime("%H%M")
    base_name = f"{date}_{time}"
    log_path = f"./logs/{base_name}_realsense_log.txt"
    color_video_path = f"{rgb_dir}/{base_name}_color.mp4"
    # depth_preview_path = f"{rgb_dir}/{base_name}_depth_preview.mp4"
    bag_path = f"{bag_dir}/{base_name}_raw.bag"

    # ---------- RealSense ----------
    frame_width, frame_height, fps = 640, 480, 30

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, fps)
    config.enable_record_to_file(bag_path)

    pipeline.start(config)
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    frame_size = (frame_width, frame_height)
    color_writer = cv2.VideoWriter(color_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    # depth_preview_writer = cv2.VideoWriter(depth_preview_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

    hand_detector = MediaPipeHandDetector()

    print("Running...  ESC 退出")

    # 行缓冲写入（减少磁盘开销）
    buf: List[str] = []
    flush_every_n = 30  # 大约 1 秒写一次（30fps）

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            while True:
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_img = np.asanyarray(color_frame.get_data())
                if color_writer.isOpened():
                    color_writer.write(color_img)

                # depth_color_frame = colorizer.colorize(depth_frame)
                # depth_colormap = np.asanyarray(depth_color_frame.get_data())
                # if depth_preview_writer.isOpened():
                #     depth_preview_writer.write(depth_colormap)

                intr = color_frame.profile.as_video_stream_profile().intrinsics

                img_xyz, world_xyz, all_valid = hand_detector.process(
                    color_img, depth_frame, intr
                )

                output = color_img.copy()
                if (len(img_xyz) == 21):
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

                if all_valid:
                    ts_ms = int(color_frame.get_timestamp())
                    buf.append(format_log_line(ts_ms, img_xyz, world_xyz))

                    if len(buf) >= flush_every_n:
                        f.writelines(buf)
                        f.flush()
                        buf.clear()

                cv2.imshow("RealSense Hands (21-point 3D)", output)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            if buf:
                f.writelines(buf)
                f.flush()

    finally:
        if color_writer.isOpened():
            color_writer.release()
        # if depth_preview_writer.isOpened():
        #     depth_preview_writer.release()
        hand_detector.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
