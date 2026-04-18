import os
import numpy as np
import cv2
import pyrealsense2 as rs

# ---------------------------------
# 从.bag 文件导出 深度视频
# 适用于 RealSense 深度相机录制的 .bag 文件
# ---------------------------------

BAG_PATH = ""
OUT_PATH = os.path.join("./video", os.path.basename(BAG_PATH).replace(".bag", "_depth.mp4"))

MAX_DIST_M = 5.0          # 只看最近 x 米
FPS_FALLBACK = 30         # bag 里取不到 fps 时用这个
USE_COLORMAP = cv2.COLORMAP_TURBO  # 比 JET 更好看


def get_stream_fps(profile, stream_type):
    """Try to infer FPS from bag stream profile; fallback if not available."""
    try:
        for sp in profile.get_streams():
            vsp = sp.as_video_stream_profile()
            if vsp and vsp.stream_type() == stream_type:
                return int(vsp.fps())
    except Exception:
        pass
    return FPS_FALLBACK


def main():
    if not os.path.exists(BAG_PATH):
        raise FileNotFoundError(BAG_PATH)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, BAG_PATH, repeat_playback=False)
    config.enable_all_streams()

    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)  # 离线尽快跑，不按实时

    # ---- RealSense 深度滤波器（很有效）----
    dec = rs.decimation_filter()        # 降采样去噪（可选）
    spat = rs.spatial_filter()          # 空间平滑（保边）
    temp = rs.temporal_filter()         # 时间平滑（抑制闪烁）
    hole = rs.hole_filling_filter()     # 补洞

    # 可微调参数：越大越平滑（也可能更糊）
    # spatial
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20)
    # temporal
    temp.set_option(rs.option.filter_smooth_alpha, 0.4)
    temp.set_option(rs.option.filter_smooth_delta, 20)

    # 拿一帧确认尺寸
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        raise RuntimeError("No depth stream found in this bag file.")

    w, h = depth_frame.get_width(), depth_frame.get_height()
    fps = get_stream_fps(profile, rs.stream.depth)
    print(f"[info] depth size = {w}x{h}, fps = {fps}")
    print(f"[info] export: {OUT_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (w, h))

    max_dist_mm = int(MAX_DIST_M * 1000)

    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # ---- RS filters ----
            # depth_frame = dec.process(depth_frame)
            # depth_frame = spat.process(depth_frame)
            # depth_frame = temp.process(depth_frame)
            # depth_frame = hole.process(depth_frame)

            depth = np.asanyarray(depth_frame.get_data())  # uint16, mm

            # ---- 只看 0~3m，并映射到 0~255 ----
            depth_clipped = np.clip(depth, 0, max_dist_mm)

            # 0m->0, 3m->255
            depth_u8 = (depth_clipped.astype(np.float32) / max_dist_mm * 255.0).astype(np.uint8)

            # 近=暖色：把“近(小值)”翻成“亮(大值)”
            # depth_u8 = 255 - depth_u8

            # ---- OpenCV 进一步去噪（对“盐胡椒”很有效）----
            depth_u8 = cv2.medianBlur(depth_u8, 5)

            # 上色
            depth_vis = cv2.applyColorMap(depth_u8, USE_COLORMAP)

            writer.write(depth_vis)

            frame_count += 1
            if frame_count % 200 == 0:
                print(f"[info] {frame_count} frames...")

    except RuntimeError:
        # 播放到 bag 末尾时，wait_for_frames 常会抛 RuntimeError
        print("[info] end of bag reached.")

    finally:
        writer.release()
        pipeline.stop()

    print(f"[done] saved {frame_count} frames to {OUT_PATH}")


if __name__ == "__main__":
    main()
