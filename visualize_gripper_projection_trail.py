#!/usr/bin/env python3
"""
夹爪投影拖影可视化（trail 版）

与 visualize_gripper_projection.py 的投影逻辑一致（包含 u/v 翻转）。
区别：
1. 对视频“每一帧”都做投影（不降采样）。
2. 每帧同时投影该帧（t）、t-10、t-20 的左右夹爪，三个点构成拖影（间隔 10 帧）。
3. 用不同颜色 / 稍小的点区分当前与过去帧。
4. 使用 parquet 中的 *state_cart_pos* 列（真实状态），而非 *cmd_cart_pos*。
5. 最终每个 episode 直接编码为一个 mp4 视频。
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import argparse

# 拖影采样偏移（帧数）：当前帧 / 10帧前 / 20帧前
TRAIL_OFFSETS = [0, 10, 20]

# 数据列
LEFT_COL = 'leftarm_state_cart_pos'
RIGHT_COL = 'rightarm_state_cart_pos'

# 默认输出目录
DEFAULT_OUTPUT_DIR = Path(
    '/mnt/vepfs01/output/klayzhou/datasets/PickPlaceEverything_Moz1/projection_trail'
)

# ============================================================
# 相机内参（1408x1280 鱼眼模型），视频分辨率 640x480
# ============================================================
ORIG_W, ORIG_H = 1408, 1280
VIDEO_W, VIDEO_H = 640, 480

FX_ORIG = 517.1607943040
FY_ORIG = 517.1787653667
CX_ORIG = 703.7844520502
CY_ORIG = 641.3050626158

K1 = 0.0919417169
K2 = -0.0033091746
K3 = -0.0143766221
K4 = 0.0034217546

SCALE_X = VIDEO_W / ORIG_W
SCALE_Y = VIDEO_H / ORIG_H

FX = FX_ORIG * SCALE_X
FY = FY_ORIG * SCALE_Y
CX = CX_ORIG * SCALE_X
CY = CY_ORIG * SCALE_Y

# ============================================================
# 相机外参：T_torso_from_cam （与主脚本一致）
# ============================================================
CAMERA_EXTRINSIC = np.array([
    [1.0,  0.0,         0.0,         0.0  ],
    [0.0, -0.37460659, -0.92718385, -72.14],
    [0.0,  0.92718385, -0.37460659,  144.02],
    [0.0,  0.0,         0.0,         1.0  ]
], dtype=np.float64)


def parse_array(val):
    if isinstance(val, np.ndarray):
        return val.astype(np.float64)
    if isinstance(val, (list, tuple)):
        return np.array(val, dtype=np.float64)
    if isinstance(val, str):
        val = val.strip().strip('[]')
        return np.array([float(x) for x in val.split()], dtype=np.float64)
    return np.array(val, dtype=np.float64)


def project_fisheye(point_torso, extrinsic, fx, fy, cx, cy, k1, k2, k3, k4):
    """与 visualize_gripper_projection.py 完全一致的投影逻辑。"""
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3] / 1000.0

    pt_torso = np.array([point_torso[0], point_torso[1], point_torso[2]])
    pt_cam = R.T @ (pt_torso - t)

    Xc, Yc, Zc = pt_cam[0], pt_cam[1], pt_cam[2]
    if Zc <= 0:
        return None

    x = Xc / Zc
    y = Yc / Zc
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(r)
    theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)

    if r > 1e-9:
        scale = theta_d / r
        xd = x * scale
        yd = y * scale
    else:
        xd, yd = x, y

    u = fx * xd + cx
    v = fy * yd + cy

    # 与主脚本一致的 u/v 翻转
    u = VIDEO_W - u
    v = VIDEO_H - v

    return (int(round(u)), int(round(v)))


def draw_point(img, px, color, radius=5, label=None):
    if px is None:
        return img
    u, v = px
    h, w = img.shape[:2]
    cv2.circle(img, (u, v), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (u, v), radius + 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    if label is not None and 0 <= u < w and 0 <= v < h:
        cv2.putText(img, label, (u + radius + 3, v + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def process_video(video_path, parquet_path, output_path):
    video_path = Path(video_path)
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'视频:   {video_path}')
    print(f'parquet: {parquet_path}')
    print(f'输出:   {output_path}')

    df = pd.read_parquet(parquet_path)
    print(f'数据行数: {len(df)}')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'视频总帧数: {total_video_frames}, 分辨率: {video_w}x{video_h}, fps: {fps:.2f}')

    n = min(total_video_frames, len(df))
    print(f'将处理 {n} 帧（视频/parquet 取较小值）')

    # 预计算所有帧的左右臂投影像素（便于做 t-10/t-20 拖影）
    left_px_cache = [None] * n
    right_px_cache = [None] * n
    for i in range(n):
        row = df.iloc[i]
        try:
            left = parse_array(row[LEFT_COL])[:3]
            right = parse_array(row[RIGHT_COL])[:3]
        except Exception:
            continue
        left_px_cache[i] = project_fisheye(left, CAMERA_EXTRINSIC,
                                           FX, FY, CX, CY, K1, K2, K3, K4)
        right_px_cache[i] = project_fisheye(right, CAMERA_EXTRINSIC,
                                            FX, FY, CX, CY, K1, K2, K3, K4)

    # 颜色（BGR），按 TRAIL_OFFSETS 顺序：索引 0 是当前帧（最亮、最大）
    left_colors = [
        (0, 255, 0),     # t    : 鲜绿
        (0, 200, 120),   # t-10 : 蓝绿
        (0, 150, 200),   # t-20 : 偏暗
    ]
    right_colors = [
        (0, 128, 255),   # t    : 橙
        (0, 90, 200),    # t-10 : 暗橙
        (64, 60, 160),   # t-20 : 暗红
    ]
    radii = [6, 4, 3]

    # 输出视频 writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (VIDEO_W, VIDEO_H))
    if not out.isOpened():
        raise RuntimeError(f'无法创建输出视频: {output_path}')

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[1] != VIDEO_W or frame.shape[0] != VIDEO_H:
            frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))

        vis = frame.copy()

        # 按 TRAIL_OFFSETS 从最久远到最近画点，保证当前帧在最上层
        # idx 在 colors/radii 中的位置固定（0 当前，1 中间，2 最远）
        for slot in reversed(range(len(TRAIL_OFFSETS))):
            offset = TRAIL_OFFSETS[slot]
            idx = i - offset
            if idx < 0:
                continue
            lp = left_px_cache[idx]
            rp = right_px_cache[idx]
            draw_point(vis, lp, left_colors[slot], radius=radii[slot],
                       label='L' if slot == 0 else None)
            draw_point(vis, rp, right_colors[slot], radius=radii[slot],
                       label='R' if slot == 0 else None)

        # 连接三点形成拖影线（从最久远 -> 当前）
        for arm_cache, color in [
            (left_px_cache, (120, 220, 120)),
            (right_px_cache, (120, 180, 220)),
        ]:
            pts = []
            for slot in reversed(range(len(TRAIL_OFFSETS))):
                offset = TRAIL_OFFSETS[slot]
                idx = i - offset
                if 0 <= idx and arm_cache[idx] is not None:
                    pts.append(arm_cache[idx])
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(vis, a, b, color, 1, lineType=cv2.LINE_AA)

        cv2.putText(vis, f'Frame:{i}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(vis,
                    f'trail offsets: {TRAIL_OFFSETS}  |  L:green R:orange',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)

        out.write(vis)

        if (i + 1) % 500 == 0 or i == n - 1:
            print(f'  处理进度: {i+1}/{n}')

    cap.release()
    out.release()
    print(f'完成，输出: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='夹爪投影拖影可视化')
    parser.add_argument('--video', required=True, help='视频路径 (mp4)')
    parser.add_argument('--parquet', default=None,
                        help='对应 parquet，不传则从视频路径自动推断')
    parser.add_argument('--output', default=None, help='输出 mp4 路径')
    args = parser.parse_args()

    video_path = Path(args.video)

    if args.parquet is None:
        parts = list(video_path.parts)
        if 'videos' in parts:
            vi = parts.index('videos')
            chunk = parts[vi + 1]
            ep_name = video_path.stem
            root = Path(*parts[:vi])
            guess = root / 'data' / chunk / f'{ep_name}.parquet'
            if guess.exists():
                args.parquet = str(guess)
            else:
                raise FileNotFoundError(
                    f'自动定位 parquet 失败: {guess}，请通过 --parquet 指定')
        else:
            raise ValueError('无法推断 parquet 路径，请使用 --parquet 指定')

    if args.output is None:
        args.output = str(DEFAULT_OUTPUT_DIR / f'{video_path.stem}_trail.mp4')

    process_video(args.video, args.parquet, args.output)


if __name__ == '__main__':
    main()
