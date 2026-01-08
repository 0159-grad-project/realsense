from datetime import datetime
from typing import List

import cv2
import numpy as np
import pyrealsense2 as rs

from mediapipe_hand import MediaPipeHandDetector, HandRecords


def format_log_line(
    ts_ns: int,
    world_xyz: HandRecords,
) -> str:
    """
    world_xyz: [(X,Y,Z), ...] length=3
    输出：timestamp_ns, X,Y,Z, ... 共 1+3*3 列
    """
    parts = [str(ts_ns)]
    for (X, Y, Z) in world_xyz:
        parts.extend(
            [
                f"{X:.16f}",
                f"{Y:.16f}",
                f"{Z:.16f}",
            ]
        )
    return ",".join(parts) + "\n"

def main():
    # ---------- RealSense ----------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # 深度滤波（稳定用）
    dec = rs.decimation_filter()  # 降采样/减小噪声（会改尺寸！）
    spat = rs.spatial_filter()  # 空间滤波
    temp = rs.temporal_filter()  # 时间滤波
    hole = rs.hole_filling_filter()  # 填充空洞

    hand_detector = MediaPipeHandDetector()

    print("Running...  ESC / q 退出")

    # 行缓冲写入（减少磁盘开销）
    buf: List[str] = []
    flush_every_n = 30  # 大约 1 秒写一次（30fps）
    date = datetime.now().strftime("%m%d")
    time = datetime.now().strftime("%H%M")
    log_path = f"./logs/{date}_{time}_realsense_log.txt"

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            while True:
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # ---------- 深度滤波 ----------
                # df = depth_frame
                # df = dec.process(df)
                # df = spat.process(df)
                # df = temp.process(df)
                # df = hole.process(df)
                # depth_frame = df.as_depth_frame()

                color_img = np.asanyarray(color_frame.get_data())

                intr = color_frame.profile.as_video_stream_profile().intrinsics

                img_xyz, world_xyz, all_valid = hand_detector.process(
                    color_img, depth_frame, intr
                )

                output = color_img.copy()
                for i, (px, py, z) in enumerate(img_xyz):
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
                    buf.append(format_log_line(ts_ms, world_xyz))

                    if len(buf) >= flush_every_n:
                        f.writelines(buf)
                        f.flush()
                        buf.clear()

                cv2.imshow("RealSense Hands (6-point 3D)", output)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            if buf:
                f.writelines(buf)
                f.flush()

    finally:
        hand_detector.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
