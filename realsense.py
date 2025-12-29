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
    """在 (x,y) 周围取邻域平均深度，返回米"""
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


def pixel_to_3d(
    intr: rs.intrinsics, x: int, y: int, z_m: float
) -> Tuple[float, float, float]:
    """像素 + 深度 → 相机坐标系 3D（米）"""
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


def main():
    # ---------- RealSense ----------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # 深度滤波（稳定用）
    dec = rs.decimation_filter()
    spat = rs.spatial_filter()
    temp = rs.temporal_filter()
    hole = rs.hole_filling_filter()

    # ---------- MediaPipe ----------
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    print("Running...  ESC / q 退出")

    last_print = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # ---------- 深度滤波并强制转回 depth_frame ----------
            df = depth_frame
            df = dec.process(df)
            df = spat.process(df)
            df = temp.process(df)
            df = hole.process(df)
            df = df.as_depth_frame()  # 🔑 必须

            color_img = np.asanyarray(color_frame.get_data())
            h, w = color_img.shape[:2]

            intr = color_frame.profile.as_video_stream_profile().intrinsics

            # ---------- MediaPipe ----------
            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            output = color_img.copy()
            hands_3d: List[Dict[str, Tuple[float, float, float]]] = []

            if result.multi_hand_landmarks:
                for hi, hand_lms in enumerate(result.multi_hand_landmarks):
                    # mp.solutions.drawing_utils.draw_landmarks(
                    #     output, hand_lms, mp_hands.HAND_CONNECTIONS
                    # )

                    one_hand = {}

                    for i, (name, lm_id) in enumerate(HAND_POINTS.items()):
                        lm = hand_lms.landmark[int(lm_id)]

                        px = clamp(int(lm.x * w), 0, w - 1)
                        py = clamp(int(lm.y * h), 0, h - 1)

                        z = depth_at_pixel_robust(df, px, py, r=2)

                        if z <= 0.0:
                            cv2.circle(output, (px, py), 4, (0, 0, 255), -1)
                            continue

                        X, Y, Z = pixel_to_3d(intr, px, py, z)
                        one_hand[name] = (X, Y, Z)

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

                    if one_hand:
                        hands_3d.append(one_hand)

            cv2.imshow("RealSense Hands (6-point 3D)", output)

            # ---------- 控制台输出 ----------
            if hands_3d:
                for i, h3d in enumerate(hands_3d):
                    ordered = {k: h3d.get(k) for k in HAND_POINTS.keys()}
                    print(f"[Hand {i+1}] {ordered}")

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        hands.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
