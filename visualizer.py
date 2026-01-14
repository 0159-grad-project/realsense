import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# ---------------------------------
# 3D 可视化 RealSense + MediaPipe 手部关键点日志
# ---------------------------------

INPUT_PATH = ""

HAND_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

HIGHLIGHT_INDICES = [0, 4, 8, 12, 16, 20]


def find_latest_log(log_dir: str) -> str:
    if not os.path.isdir(log_dir):
        return ""
    logs = [
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.lower().endswith(".txt")
    ]
    return max(logs, key=os.path.getmtime) if logs else ""


def load_log(path: str) -> Tuple[np.ndarray, np.ndarray]:
    expected_len = 1 + 21 * 6
    timestamps: List[int] = []
    frames: List[List[Tuple[float, float, float]]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < expected_len:
                continue
            try:
                ts = int(float(parts[0]))
                coords: List[Tuple[float, float, float]] = []
                idx = 1
                invalid_line = False
                for _ in range(21):
                    z_m = float(parts[idx + 2])
                    idx += 3  # px, py, z_m
                    X = float(parts[idx])
                    Y = float(parts[idx + 1])
                    Z = float(parts[idx + 2])
                    if z_m == 0.0 or Z == 0.0:
                        invalid_line = True
                    coords.append((X, Y, Z))
                    idx += 3
            except ValueError:
                continue

            if invalid_line:
                continue

            timestamps.append(ts)
            frames.append(coords)

    if not frames:
        raise SystemExit("No valid frames found in log.")

    return np.asarray(timestamps, dtype=np.int64), np.asarray(frames, dtype=np.float32)


def set_axes_limits(
    ax,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    pad_ratio: float = 0.1,
    min_pad: float = 10.0,
) -> None:
    ranges = max_xyz - min_xyz
    pad = np.maximum(ranges * pad_ratio, min_pad)
    ax.set_xlim(min_xyz[0] - pad[0], max_xyz[0] + pad[0])
    ax.set_ylim(min_xyz[1] - pad[1], max_xyz[1] + pad[1])
    ax.set_zlim(min_xyz[2] - pad[2], max_xyz[2] + pad[2])


def main() -> None:
    log_path = INPUT_PATH or find_latest_log("./logs_bag") or find_latest_log("./logs")
    if not log_path:
        raise SystemExit("No log file found. Provide a log path or put logs in ./logs_bag.")

    timestamps, points = load_log(log_path)
    points = points * 1000.0
    time_ms = timestamps
    min_xyz = points.min(axis=(0, 1))
    max_xyz = points.max(axis=(0, 1))

    fig = plt.figure()
    plt.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=20, azim=-60)
    set_axes_limits(ax, min_xyz, max_xyz)

    pts0 = points[0]
    scatter = ax.scatter(pts0[:, 0], pts0[:, 1], pts0[:, 2], c="tab:blue", s=18)
    highlight = ax.scatter(
        pts0[HIGHLIGHT_INDICES, 0],
        pts0[HIGHLIGHT_INDICES, 1],
        pts0[HIGHLIGHT_INDICES, 2],
        c="tab:red",
        s=32,
    )
    lines = []
    for i, j in HAND_EDGES:
        line, = ax.plot(
            [pts0[i, 0], pts0[j, 0]],
            [pts0[i, 1], pts0[j, 1]],
            [pts0[i, 2], pts0[j, 2]],
            c="tab:orange",
            lw=2,
        )
        lines.append(line)

    time_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes)

    def draw_frame(frame_idx: int) -> None:
        pts = points[frame_idx]
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        highlight._offsets3d = (
            pts[HIGHLIGHT_INDICES, 0],
            pts[HIGHLIGHT_INDICES, 1],
            pts[HIGHLIGHT_INDICES, 2],
        )
        for line, (i, j) in zip(lines, HAND_EDGES):
            line.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])
            line.set_3d_properties([pts[i, 2], pts[j, 2]])
        time_text.set_text(f"t = {time_ms[frame_idx]} ms")

    ax_slider = fig.add_axes([0.16, 0.08, 0.68, 0.03])
    slider = Slider(
        ax_slider,
        "t (ms)",
        float(time_ms[0]),
        float(time_ms[-1]),
        valinit=float(time_ms[0]),
        valstep=time_ms.astype(float),
    )

    current_idx = 0
    updating = {"slider": False}

    def set_frame(frame_idx: int, update_slider: bool = False) -> None:
        nonlocal current_idx
        frame_idx = max(0, min(frame_idx, len(points) - 1))
        current_idx = frame_idx
        draw_frame(frame_idx)
        if update_slider:
            updating["slider"] = True
            slider.set_val(float(time_ms[frame_idx]))
            updating["slider"] = False
        fig.canvas.draw_idle()

    def on_slider_change(val):
        if updating["slider"]:
            return
        frame_idx = int(np.searchsorted(time_ms, val, side="left"))
        set_frame(frame_idx)

    slider.on_changed(on_slider_change)

    ax_prev = fig.add_axes([0.06, 0.08, 0.05, 0.03])
    ax_next = fig.add_axes([0.89, 0.08, 0.05, 0.03])
    btn_prev = Button(ax_prev, "<")
    btn_next = Button(ax_next, ">")

    def on_prev(_event):
        set_frame(current_idx - 1, update_slider=True)

    def on_next(_event):
        set_frame(current_idx + 1, update_slider=True)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    set_frame(0, update_slider=True)
    plt.title(os.path.basename(log_path))
    plt.show()


if __name__ == "__main__":
    main()
