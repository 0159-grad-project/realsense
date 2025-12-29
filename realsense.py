from datetime import datetime
from typing import Dict, Tuple, List

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def depth_at_pixel_robust(
    depth_frame: rs.depth_frame,
    x: int,
    y: int,
    r: int = 2,
    min_valid: float = 1e-6,
    max_valid: float = 10.0,
) -> float:
    """在 (x,y) 周围取5x5邻域平均深度，返回米"""
    vp = depth_frame.profile.as_video_stream_profile()
    w = vp.width()
    h = vp.height()

    vals = []
    for dy in range(-r, r + 1):
        yy = clamp(y + dy, 0, h - 1)
        for dx in range(-r, r + 1):
            xx = clamp(x + dx, 0, w - 1)
            d = depth_frame.get_distance(xx, yy)
            if min_valid < d < max_valid:
                vals.append(d)

    return sum(vals) / len(vals) if vals else 0.0


# 把图片上的像素坐标和深度值转换为相机坐标系下的 3D 坐标（米）
def pixel_to_3d(
    intr: rs.intrinsics, x: int, y: int, z_m: float
) -> Tuple[float, float, float]:
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], z_m)
    return float(X), float(Y), float(Z)


# MediaPipe 手部关键点定义
mp_hands = mp.solutions.hands
HAND_POINTS = {
    "wrist": mp_hands.HandLandmark.WRIST,
    "thumb": mp_hands.HandLandmark.THUMB_TIP,
    "index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
    "middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    "ring": mp_hands.HandLandmark.RING_FINGER_TIP,
    "pinky": mp_hands.HandLandmark.PINKY_TIP,
}

def format_log_line(
    ts_ns: int,
    records: List[Tuple[int, int, float, float, float, float]],
) -> str:
    """
    records: [(px,py,z,X,Y,Z), ...] length=6
    输出：timestamp_ns, px,py,z,X,Y,Z, ... 共 1+6*6 列
    """
    parts = [str(ts_ns)]
    for (px, py, z, X, Y, Z) in records:
        parts.extend([
            str(int(px)),
            str(int(py)),
            f"{z:.16f}",
            f"{X:.16f}",
            f"{Y:.16f}",
            f"{Z:.16f}",
        ])
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
    dec = rs.decimation_filter()  # 降采样/减小噪声
    spat = rs.spatial_filter()  # 空间滤波
    temp = rs.temporal_filter()  # 时间滤波
    hole = rs.hole_filling_filter()  # 填充空洞

    # ---------- MediaPipe Hands 初始化 ----------
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    print("Running...  ESC / q 退出")

    # 行缓冲写入（减少磁盘开销）
    buf: List[str] = []
    flush_every_n = 30  # 大约 1 秒写一次（30fps）
    today = datetime.now().strftime("%Y%m%d")
    log_path = f"./logs/{today}_realsense_log.txt"

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
                df = depth_frame
                df = dec.process(df)
                df = spat.process(df)
                df = temp.process(df)
                df = hole.process(df)
                depth_frame = df.as_depth_frame()

                color_img = np.asanyarray(color_frame.get_data())
                h, w = color_img.shape[:2]

                intr = color_frame.profile.as_video_stream_profile().intrinsics

                # ---------- MediaPipe ----------
                rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                output = color_img.copy()

                if result.multi_hand_landmarks:
                    for hi, hand_lms in enumerate(result.multi_hand_landmarks):
                        # mp.solutions.drawing_utils.draw_landmarks(
                        #     output, hand_lms, mp_hands.HAND_CONNECTIONS
                        # )

                        records: List[Tuple[int, int, float, float, float, float]] = []
                        all_valid = True

                        for i, (name, lm_id) in enumerate(HAND_POINTS.items()):
                            lm = hand_lms.landmark[int(lm_id)]

                            px = clamp(int(lm.x * w), 0, w - 1)
                            py = clamp(int(lm.y * h), 0, h - 1)

                            z = depth_at_pixel_robust(depth_frame, px, py, r=2)

                            if z <= 0.0:
                                all_valid = False
                                cv2.circle(output, (px, py), 4, (0, 0, 255), -1)
                                cv2.putText(
                                    output, f"{i}", (px + 6, py - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1
                                )
                                continue

                            X, Y, Z = pixel_to_3d(intr, px, py, z)
                            records.append((px, py, z, X, Y, Z))

                            cv2.circle(output, (px, py), 4, (0, 255, 0), -1)
                            cv2.putText(
                                output,
                                f"{i}",  # f"{name}: {Z:.2f}m",
                                (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 255, 0),
                                1,
                            )
                        
                    if all_valid and len(records) == 6:
                        ts_ns = int(color_frame.get_timestamp() * 1_000_000)
                        buf.append(format_log_line(ts_ns, records))

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
        hands.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
