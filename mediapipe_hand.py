from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

# ---------------------------------
# MediaPipe 手部关键点检测 + RealSense 深度相机
# 获取手部 21 个关键点的像素坐标、深度（米）和空间坐标（X,Y,Z，米）
# ---------------------------------


HandRecord = Tuple[int, int, float, float, float, float]  # px, py, z_m, X, Y, Z
HandRecords = List[HandRecord]
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

# HAND_POINTS = {
#     "wrist": mp_hands.HandLandmark.WRIST,
#     "thumb": mp_hands.HandLandmark.THUMB_TIP,
#     "index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
#     "middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
#     "ring": mp_hands.HandLandmark.RING_FINGER_TIP,
#     "pinky": mp_hands.HandLandmark.PINKY_TIP,
# }


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

    @staticmethod
    def _select_labeled(hands: List["HandData"], label_prefix: str) -> Optional["HandData"]:
        prefix = label_prefix.lower()
        candidates = [hand for hand in hands if hand.label.lower().startswith(prefix)]
        if not candidates:
            return None
        return max(candidates, key=lambda hand: hand.score)

    @staticmethod
    def _assign_by_position(
        hands: List["HandData"],
        left: Optional["HandData"],
        right: Optional["HandData"],
        img_w: int,
    ) -> Tuple[Optional["HandData"], Optional["HandData"]]:
        if not hands:
            return left, right

        if len(hands) == 1:
            if left is not None or right is not None:
                return left, right
            only = hands[0]
            if only.avg_x <= img_w * 0.5:
                return only, None
            return None, only

        ordered = sorted(hands, key=lambda hand: hand.avg_x)
        if left is None and right is None:
            return ordered[0], ordered[-1]

        if left is None:
            for hand in ordered:
                if hand is not right:
                    left = hand
                    break

        if right is None:
            for hand in reversed(ordered):
                if hand is not left:
                    right = hand
                    break

        return left, right

    def _extract_points(
        self,
        hand_lms,
        depth_frame: rs.depth_frame,
        intr: rs.intrinsics,
        img_h: int,
        img_w: int,
    ) -> Tuple[HandRecords, bool]:
        records: HandRecords = []
        all_valid = True
        depths: List[float] = []

        for lm_id in HAND_LANDMARKS:
            lm = hand_lms.landmark[int(lm_id)]

            px = clamp(int(lm.x * img_w), 0, img_w - 1)
            py = clamp(int(lm.y * img_h), 0, img_h - 1)

            # z = depth_at_pixel_robust(depth_frame, px, py, r=1)
            z = depth_frame.get_distance(px, py)
            depths.append(z)

            if z <= 0.0:
                all_valid = False

            X, Y, Z = pixel_to_3d(intr, px, py, z)
            records.append((px, py, z, X, Y, Z))

        if depths and (max(depths) - min(depths)) > 0.3:
            all_valid = False

        return records, bool(all_valid and len(records) == 21)

    def process(
        self,
        color_img: np.ndarray,
        depth_frame: rs.depth_frame,
        intr: rs.intrinsics,
    ) -> Tuple[Optional["HandData"], Optional["HandData"]]:
        """Run MediaPipe on a color frame and return left/right hand data if present."""
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None, None

        img_h, img_w = color_img.shape[:2]
        hands: List["HandData"] = []
        handedness = result.multi_handedness or []

        for idx, hand_lms in enumerate(result.multi_hand_landmarks):
            label = ""
            score = 0.0
            if idx < len(handedness) and handedness[idx].classification:
                cls = handedness[idx].classification[0]
                label = cls.label or ""
                score = float(cls.score or 0.0)

            records, valid = self._extract_points(hand_lms, depth_frame, intr, img_h, img_w)
            avg_x = sum(record[0] for record in records) / len(records) if records else 0.0
            hands.append(
                HandData(
                    label=label,
                    score=score,
                    records=records,
                    valid=valid,
                    avg_x=avg_x,
                )
            )

        left = self._select_labeled(hands, "left")
        right = self._select_labeled(hands, "right")
        if left is None or right is None:
            left, right = self._assign_by_position(hands, left, right, img_w)

        return left, right


@dataclass(frozen=True)
class HandData:
    label: str
    score: float
    records: HandRecords
    valid: bool
    avg_x: float
