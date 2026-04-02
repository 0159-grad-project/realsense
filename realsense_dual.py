import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, TextIO

import cv2
import numpy as np
import pyrealsense2 as rs

from mediapipe_hand import MediaPipeHandDetector
from realsense import (
    HAND_DRAW_INDICES,
    LEFT_COLOR,
    RIGHT_DRAW_INDICES,
    RIGHT_COLOR,
    _merge_hands as merge_hands,
    format_log_line,
)

# ---------------------------------
# Dual RealSense D435i keypoint logging with MediaPipe Hands
# Saves one log per camera into ./logs_2 on each run.
# ---------------------------------

ENABLE_SAVING = False  # Save .bag files in addition to logs
MAX_NUM_HANDS = 2  # 1 or 2

FRAME_WIDTH = 1280  # 640
FRAME_HEIGHT = 720  # 480
FPS = 30
FRAME_WAIT_TIMEOUT_MS = 5

# Leave empty to auto-select two connected D435i devices.
# Otherwise set to two serial numbers in the desired order.
DEVICE_SERIALS: List[str] = []

LOG_ROOT = "./logs_2"


def _select_serials() -> List[str]:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if len(devices) < 2:
        raise SystemExit("Need at least two RealSense devices connected.")

    available_serials = [
        dev.get_info(rs.camera_info.serial_number) for dev in devices
    ]

    if DEVICE_SERIALS:
        missing = [s for s in DEVICE_SERIALS if s not in available_serials]
        if missing:
            raise SystemExit(f"Missing devices: {', '.join(missing)}")
        return DEVICE_SERIALS[:2]

    d435i_serials = []
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        if "d435i" in name.lower():
            d435i_serials.append(dev.get_info(rs.camera_info.serial_number))

    if len(d435i_serials) >= 2:
        return d435i_serials[:2]

    return available_serials[:2]


@dataclass
class CameraSession:
    serial: str
    pipeline: rs.pipeline
    align: rs.align
    hand_detector: MediaPipeHandDetector
    log_file: TextIO
    buffer: List[str]
    window_name: str


def _start_camera(
    serial: str,
    idx: int,
    base_name: str,
    bag_dir: str,
    log_root: str,
) -> CameraSession:
    log_path = os.path.join(
        log_root, f"{base_name}_cam{idx + 1}_realsense_log.txt"
    )
    bag_path = os.path.join(bag_dir, f"{base_name}_cam{idx + 1}.bag")

    log_file = open(log_path, "w", encoding="utf-8")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS
    )
    config.enable_stream(
        rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS
    )
    if ENABLE_SAVING:
        config.enable_record_to_file(bag_path)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    hand_detector = MediaPipeHandDetector(max_num_hands=MAX_NUM_HANDS)
    window_name = f"RealSense Cam{idx + 1} ({serial})"

    return CameraSession(
        serial=serial,
        pipeline=pipeline,
        align=align,
        hand_detector=hand_detector,
        log_file=log_file,
        buffer=[],
        window_name=window_name,
    )


def main() -> None:
    os.makedirs(LOG_ROOT, exist_ok=True)
    bag_dir = os.path.join(LOG_ROOT, "raw_bags")
    if ENABLE_SAVING:
        os.makedirs(bag_dir, exist_ok=True)

    serials = _select_serials()
    print(f"Using devices: {', '.join(serials)}")

    date = datetime.now().strftime("%m%d")
    time = datetime.now().strftime("%H%M")
    base_name = f"{date}_{time}"

    sessions: List[CameraSession] = []
    try:
        for idx, serial in enumerate(serials):
            sessions.append(_start_camera(serial, idx, base_name, bag_dir, LOG_ROOT))

        print("Running...  ESC/q to quit")
        flush_every_n = 30

        while True:
            for session in sessions:
                has_frames, frames = session.pipeline.try_wait_for_frames(
                    FRAME_WAIT_TIMEOUT_MS
                )
                if not has_frames:
                    continue
                frames = session.align.process(frames)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_img = np.asanyarray(color_frame.get_data())

                intr = color_frame.profile.as_video_stream_profile().intrinsics

                # Swap left/right due to mirrored view
                right_hand, left_hand = session.hand_detector.process(
                    color_img, depth_frame, intr
                )
                ts_ms = int(color_frame.get_timestamp())

                output = color_img.copy()
                if MAX_NUM_HANDS == 2:
                    records, left_valid, right_valid = merge_hands(
                        left_hand, right_hand
                    )
                    for indices, color in (
                        (HAND_DRAW_INDICES, LEFT_COLOR),
                        (RIGHT_DRAW_INDICES, RIGHT_COLOR),
                    ):
                        for i in indices:
                            if records[i] is not None:
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
                    if left_valid or right_valid:
                        session.buffer.append(format_log_line(ts_ms, records))

                        if len(session.buffer) >= flush_every_n:
                            session.log_file.writelines(session.buffer)
                            session.log_file.flush()
                            session.buffer.clear()
                else:
                    hand = left_hand if left_hand is not None else right_hand
                    if hand is not None:
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

                        if hand.valid:
                            session.buffer.append(format_log_line(ts_ms, records))

                            if len(session.buffer) >= flush_every_n:
                                session.log_file.writelines(session.buffer)
                                session.log_file.flush()
                                session.buffer.clear()

                cv2.imshow(session.window_name, output)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        for session in sessions:
            if session.buffer:
                session.log_file.writelines(session.buffer)
                session.log_file.flush()
                session.buffer.clear()

    finally:
        for session in sessions:
            session.hand_detector.close()
            session.pipeline.stop()
            try:
                session.log_file.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
