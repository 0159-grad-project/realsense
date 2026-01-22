import os
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from mediapipe_hand import HandData, HandRecords, MediaPipeHandDetector

# ---------------------------------
# RealSense 深度相机数据记录 + MediaPipe 手部关键点检测
# 记录 21 个关键点的像素坐标、深度（米）和空间坐标（X,Y,Z）
# 同时保存 RGB 视频和 .bag 文件
# ---------------------------------

ENABLE_RECORDING = True  # 是否保存视频和 .bag 文件
ENABLE_PUB = False  # 是否启用 ZeroMQ
MAX_NUM_HANDS = 2  # 检测手的最大数量（1 或 2）

# ZeroMQ 发布配置
PUB_PORT = 5555
PUB_TOPIC = b"realsense"
HAND_DRAW_INDICES = [0, 4, 8, 12, 16, 20]
RIGHT_DRAW_INDICES = [i + 21 for i in HAND_DRAW_INDICES]
RS_SEND_INDICES_ONE = HAND_DRAW_INDICES
RS_SEND_INDICES_TWO = HAND_DRAW_INDICES + RIGHT_DRAW_INDICES
RS_SEND_INDICES = RS_SEND_INDICES_TWO if MAX_NUM_HANDS == 2 else RS_SEND_INDICES_ONE
RS_SEND_SCALE = 1000.0

LEFT_COLOR = (0, 255, 0)
RIGHT_COLOR = (255, 0, 0)

if ENABLE_PUB:
    import msgpack
    import zmq


def format_log_line(
    ts_ms: int,
    records: HandRecords,
) -> str:
    """
    records: [(px, py, z_m, X, Y, Z), ...] length=21 or 42
    输出：timestamp_ms, px,py,z_m,X,Y,Z, ... 共 1 + N*6 列
    """
    parts = [str(ts_ms)]
    for px, py, z, X, Y, Z in records:
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


def _empty_hand_points() -> HandRecords:
    return [(0, 0, 0.0, 0.0, 0.0, 0.0)] * 21


def _merge_hands(
    left: Optional[HandData],
    right: Optional[HandData],
) -> Tuple[HandRecords, bool, bool]:
    left_valid = bool(left is not None and left.valid)
    right_valid = bool(right is not None and right.valid)

    left_records = _empty_hand_points()
    right_records = _empty_hand_points()

    if left_valid:
        left_records = left.records
    if right_valid:
        right_records = right.records

    records = left_records + right_records
    return records, left_valid, right_valid


def main():
    rgb_dir = f"./logs/rgb_videos"
    bag_dir = f"./logs/raw_bags"

    date = datetime.now().strftime("%m%d")
    time = datetime.now().strftime("%H%M")
    base_name = f"{date}_{time}"
    log_path = f"./logs/{base_name}_realsense_log.txt"
    color_video_path = f"{rgb_dir}/{base_name}_color.mp4"
    # depth_preview_path = f"{rgb_dir}/{base_name}_depth_preview.mp4"
    bag_path = f"{bag_dir}/{base_name}.bag"

    # ---------- RealSense ----------
    frame_width, frame_height, fps = 640, 480, 30

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, fps)
    if ENABLE_RECORDING:
        config.enable_record_to_file(bag_path)

    pipeline.start(config)
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    frame_size = (frame_width, frame_height)
    color_writer = None
    if ENABLE_RECORDING:
        color_writer = cv2.VideoWriter(
            color_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
        )
        # depth_preview_writer = cv2.VideoWriter(depth_preview_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

    hand_detector = MediaPipeHandDetector(max_num_hands=MAX_NUM_HANDS)

    pub = None
    pub_ctx = None
    if ENABLE_PUB:
        pub_ctx = zmq.Context.instance()
        pub = pub_ctx.socket(zmq.PUB)
        pub.bind(f"tcp://*:{PUB_PORT}")

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
                if color_writer is not None and color_writer.isOpened():
                    color_writer.write(color_img)

                # depth_color_frame = colorizer.colorize(depth_frame)
                # depth_colormap = np.asanyarray(depth_color_frame.get_data())
                # if depth_preview_writer.isOpened():
                #     depth_preview_writer.write(depth_colormap)

                intr = color_frame.profile.as_video_stream_profile().intrinsics

                left_hand, right_hand = hand_detector.process(color_img, depth_frame, intr)
                ts_ms = int(color_frame.get_timestamp())

                output = color_img.copy()
                if MAX_NUM_HANDS == 2:
                    records, left_valid, right_valid = _merge_hands(
                        left_hand, right_hand
                    )
                    has_valid = left_valid or right_valid
                    if has_valid:
                        for indices, color, enabled in (
                            (HAND_DRAW_INDICES, LEFT_COLOR, left_valid),
                            (RIGHT_DRAW_INDICES, RIGHT_COLOR, right_valid),
                        ):
                            if not enabled:
                                continue
                            for i in indices:
                                px, py, z, _, _, _ = records[i]
                                draw_color = color if z > 0 else (0, 0, 255)
                                cv2.circle(output, (px, py), 4, draw_color, -1)
                                cv2.putText(
                                    output,
                                    f"{z:.2f}",
                                    (px + 6, py - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45,
                                    draw_color,
                                    1,
                                )

                        buf.append(format_log_line(ts_ms, records))

                        if len(buf) >= flush_every_n:
                            f.writelines(buf)
                            f.flush()
                            buf.clear()

                        if ENABLE_PUB and pub is not None:
                            send_pts = [
                                [
                                    records[i][3] * RS_SEND_SCALE,
                                    records[i][4] * RS_SEND_SCALE,
                                    records[i][5] * RS_SEND_SCALE,
                                ]
                                for i in RS_SEND_INDICES
                            ]
                            payload = {
                                "src": "realsense",
                                "markers": send_pts,
                                "timestamp": ts_ms,
                            }
                            pub.send_multipart(
                                [PUB_TOPIC, msgpack.packb(payload, use_bin_type=True)]
                            )
                else:
                    hand = left_hand if left_hand is not None else right_hand
                    if hand is not None and hand.valid:
                        records = hand.records
                        for i in HAND_DRAW_INDICES:
                            px, py, z, _, _, _ = records[i]
                            draw_color = LEFT_COLOR if z > 0 else (0, 0, 255)
                            cv2.circle(output, (px, py), 4, draw_color, -1)
                            cv2.putText(
                                output,
                                f"{z:.2f}",
                                (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                draw_color,
                                1,
                            )

                        buf.append(format_log_line(ts_ms, records))

                        if len(buf) >= flush_every_n:
                            f.writelines(buf)
                            f.flush()
                            buf.clear()

                        if ENABLE_PUB and pub is not None:
                            send_pts = [
                                [
                                    records[i][3] * RS_SEND_SCALE,
                                    records[i][4] * RS_SEND_SCALE,
                                    records[i][5] * RS_SEND_SCALE,
                                ]
                                for i in RS_SEND_INDICES
                            ]
                            payload = {
                                "src": "realsense",
                                "markers": send_pts,
                                "timestamp": ts_ms,
                            }
                            pub.send_multipart(
                                [PUB_TOPIC, msgpack.packb(payload, use_bin_type=True)]
                            )

                cv2.imshow("RealSense Hands (21-point 3D)", output)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            if buf:
                f.writelines(buf)
                f.flush()

    finally:
        if color_writer is not None and color_writer.isOpened():
            color_writer.release()
        # if depth_preview_writer.isOpened():
        #     depth_preview_writer.release()
        hand_detector.close()
        pipeline.stop()
        if pub is not None:
            pub.close(0)
        if pub_ctx is not None:
            pub_ctx.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
