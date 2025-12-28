import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 1) 配置并启动
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # 2) 以后把所有帧都对齐到彩色相机的坐标系
    align = rs.align(rs.stream.color)

    # 3) 用于把深度做成“伪彩色”显示
    colorizer = rs.colorizer()

    # 4) 取 depth scale（把 z16 转成米会用到）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth scale (meters per unit):", depth_scale)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # 对齐
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 转 numpy
            color_image = np.asanyarray(color_frame.get_data())

            # 伪彩色深度（Viewer风格）
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_colormap = np.asanyarray(depth_color_frame.get_data())

            # 读中心点距离（单位：米）
            h, w = color_image.shape[:2]
            cx, cy = w // 2, h // 2
            dist_m = depth_frame.get_distance(cx, cy)

            # 在 RGB 上画十字和距离
            cv2.drawMarker(color_image, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(color_image, f"{dist_m:.3f} m", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 拼起来显示：左RGB右Depth
            show = np.hstack((color_image, depth_colormap))
            cv2.imshow("RealSense RGB | Depth", show)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC 或 q 退出
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
