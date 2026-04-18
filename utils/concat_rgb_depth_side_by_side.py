import cv2
import os

# ---------------------------------
# 将 RGB 视频和深度视频左右拼接成一个新视频，方便对齐查看
# 适用于 RealSense 深度相机录制的 .bag 文件导出的 RGB 和深度视频
# ---------------------------------

RGB_PATH = ""     # RGB视频路径
DEPTH_PATH = ""     # 深度视频路径
OUT_PATH = ""     # 输出视频路径

# 输出宽度 = 左右各自按高度等比缩放后的宽度之和
# 输出高度：用两者中较小的高度，避免放大造成糊/噪点
USE_MIN_HEIGHT = True

# 编码器：Windows上 mp4v 通常可用；如果打不开/写失败，改成 avi+MJPG
FOURCC = "mp4v"   # or "MJPG" with .avi


def get_video_info(cap: cv2.VideoCapture):
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, w, h, n


def resize_to_height(frame, target_h):
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    new_w = int(w * (target_h / h))
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def main():
    if not os.path.exists(RGB_PATH):
        raise FileNotFoundError(RGB_PATH)
    if not os.path.exists(DEPTH_PATH):
        raise FileNotFoundError(DEPTH_PATH)

    out_dir = os.path.dirname(OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cap_rgb = cv2.VideoCapture(RGB_PATH)
    cap_dep = cv2.VideoCapture(DEPTH_PATH)

    if not cap_rgb.isOpened():
        raise RuntimeError(f"Failed to open RGB video: {RGB_PATH}")
    if not cap_dep.isOpened():
        raise RuntimeError(f"Failed to open DEPTH video: {DEPTH_PATH}")

    fps_rgb, w_rgb, h_rgb, n_rgb = get_video_info(cap_rgb)
    fps_dep, w_dep, h_dep, n_dep = get_video_info(cap_dep)

    # 输出 fps：取两者较小的（更不容易“加速/变慢”）
    out_fps = fps_rgb if fps_rgb > 0 else 30
    if fps_dep > 0:
        out_fps = min(out_fps, fps_dep)

    # 输出高度：默认用两者较小高度（不放大）
    if USE_MIN_HEIGHT:
        out_h = min(h_rgb, h_dep)
    else:
        out_h = max(h_rgb, h_dep)

    # 试读一帧来确定最终输出宽度
    ok1, fr1 = cap_rgb.read()
    ok2, fr2 = cap_dep.read()
    if not ok1 or not ok2:
        raise RuntimeError("Failed to read first frames from videos.")

    fr1 = resize_to_height(fr1, out_h)
    fr2 = resize_to_height(fr2, out_h)
    out_w = fr1.shape[1] + fr2.shape[1]

    # 复位到开头
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_dep.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 写出
    if FOURCC.lower() == "mjpg" and OUT_PATH.lower().endswith(".mp4"):
        raise RuntimeError("MJPG usually should be used with .avi output. Change OUT_PATH to .avi")

    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(OUT_PATH, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open. Try OUT_PATH=.avi and FOURCC='MJPG'.")

    print("[info] RGB:", (w_rgb, h_rgb), "fps=", fps_rgb, "frames=", n_rgb)
    print("[info] DEP:", (w_dep, h_dep), "fps=", fps_dep, "frames=", n_dep)
    print("[info] OUT:", (out_w, out_h), "fps=", out_fps)
    print("[info] writing:", OUT_PATH)

    frame_count = 0
    while True:
        ok1, rgb = cap_rgb.read()
        ok2, dep = cap_dep.read()
        if not ok1 or not ok2:
            break  # 剪到共同长度：哪个先结束就停

        rgb = resize_to_height(rgb, out_h)
        dep = resize_to_height(dep, out_h)

        combo = cv2.hconcat([rgb, dep])  # 左右拼接
        if combo.shape[1] != out_w or combo.shape[0] != out_h:
            combo = cv2.resize(combo, (out_w, out_h), interpolation=cv2.INTER_AREA)

        writer.write(combo)
        frame_count += 1

        if frame_count % 200 == 0:
            print(f"[info] {frame_count} frames...")

    writer.release()
    cap_rgb.release()
    cap_dep.release()

    print(f"[done] wrote {frame_count} frames to {OUT_PATH}")


if __name__ == "__main__":
    main()
