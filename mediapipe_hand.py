from typing import List, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp


HandRecord = Tuple[float, float, float]
HandRecords = List[HandRecord]
PixelDepths = List[Tuple[int, int, float]]
LandmarkList = List[mp.solutions.hands.HandLandmark]


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
    """Average depth (meters) in a square neighborhood around (x, y)."""
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


def pixel_to_3d(intr: rs.intrinsics, x: int, y: int, z_m: float) -> Tuple[float, float, float]:
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], z_m)
    return float(X), float(Y), float(Z)


mp_hands = mp.solutions.hands
HAND_LANDMARKS: LandmarkList = list(mp_hands.HandLandmark)
NUM_LANDMARKS = len(HAND_LANDMARKS)

HAND_POINTS = {
    "wrist": mp_hands.HandLandmark.WRIST,
    "thumb": mp_hands.HandLandmark.THUMB_TIP,
    "index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
    "middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    "ring": mp_hands.HandLandmark.RING_FINGER_TIP,
    "pinky": mp_hands.HandLandmark.PINKY_TIP,
}


class MediaPipeHandDetector:
    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def process(
        self,
        color_img: np.ndarray,
        depth_frame: rs.depth_frame,
        intr: rs.intrinsics,
    ) -> Tuple[PixelDepths, HandRecords, bool]:
        """Run MediaPipe on a color frame and return per-point depth and 3D coords."""
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        h, w = color_img.shape[:2]  # (480, 640, 3)
        img_xyz: PixelDepths = []
        world_xyz: HandRecords = []
        all_valid = True

        if not result.multi_hand_landmarks:
            return img_xyz, world_xyz, False

        hand_lms = result.multi_hand_landmarks[0]

        for lm_id in HAND_LANDMARKS:
            lm = hand_lms.landmark[int(lm_id)]

            px = clamp(int(lm.x * w), 0, w - 1)
            py = clamp(int(lm.y * h), 0, h - 1)

            # z = depth_at_pixel_robust(depth_frame, px, py, r=1)
            # print(px, py, depth_frame.get_width(), depth_frame.get_height())
            z = depth_frame.get_distance(px, py)
            img_xyz.append((px, py, z))

            if z <= 0.0:
                all_valid = False

            X, Y, Z = pixel_to_3d(intr, px, py, z)
            world_xyz.append((X, Y, Z))

        return img_xyz, world_xyz, bool(all_valid and len(world_xyz) == NUM_LANDMARKS)
