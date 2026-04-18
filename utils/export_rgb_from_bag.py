import os
import numpy as np
import cv2
import pyrealsense2 as rs

# ---------------------------------
# 从.bag 文件导出 RGB 视频
# 适用于 RealSense 深度相机录制的 .bag 文件
# ---------------------------------

BAG_PATH = ""
OUT_PATH = os.path.join("./video", os.path.basename(BAG_PATH).replace(".bag", "_rgb.mp4"))

FOURCC = "mp4v"
FPS_FALLBACK = 30


def get_stream_fps(profile, stream_type):
    try:
        for sp in profile.get_streams():
            vsp = sp.as_video_stream_profile()
            if vsp and vsp.stream_type() == stream_type:
                return int(vsp.fps())
    except Exception:
        pass
    return FPS_FALLBACK


def color_frame_to_bgr(color_frame) -> np.ndarray:
    """Convert RealSense color frame to BGR uint8 image for OpenCV."""
    fmt = color_frame.get_profile().format()
    img = np.asanyarray(color_frame.get_data())

    if fmt == rs.format.bgr8:
        return img

    if fmt == rs.format.rgb8:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if fmt == rs.format.rgba8:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return bgr

    if fmt == rs.format.bgra8:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return bgr

    if fmt == rs.format.yuyv:
        # YUYV (YUY2) -> BGR
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUY2)

    if fmt == rs.format.uyvy:
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)

    # 兜底：如果是奇怪格式，尽量当作 3 通道处理
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise RuntimeError(f"Unsupported color format: {fmt}")


def main():
    if not os.path.exists(BAG_PATH):
        raise FileNotFoundError(BAG_PATH)

    out_dir = os.path.dirname(OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, BAG_PATH, repeat_playback=False)
    config.enable_all_streams()

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("No color stream found in this bag file.")

    w, h = color_frame.get_width(), color_frame.get_height()
    fps = get_stream_fps(profile, rs.stream.color)
    fmt = color_frame.get_profile().format()

    print(f"[info] color size = {w}x{h}, fps = {fps}, format = {fmt}")
    print(f"[info] export: {OUT_PATH} ({FOURCC})")

    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open (codec unsupported).")

    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            bgr = color_frame_to_bgr(color_frame)

            # 尺寸保护
            if bgr.shape[1] != w or bgr.shape[0] != h:
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

            writer.write(bgr)

            frame_count += 1
            if frame_count % 200 == 0:
                print(f"[info] {frame_count} frames...")

    except RuntimeError:
        print("[info] end of bag reached.")

    finally:
        writer.release()
        pipeline.stop()

    print(f"[done] saved {frame_count} frames to {OUT_PATH}")


if __name__ == "__main__":
    main()
