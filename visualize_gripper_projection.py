#!/usr/bin/env python3
"""
夹爪投影可视化工具

相机信息：
- 内参：camera_calibration.txt（鱼眼模型，1408x1280分辨率）
- 外参：相机相对于 torso_state_cart_pos 的变换矩阵（平移单位为毫米）
- 视频分辨率：640x480（原始1408x1280 resize后），需要缩放内参

数据说明：
- leftarm_state_cart_pos: 左臂末端在 torso 坐标系中的位置 [x, y, z, rx, ry, rz]（单位：米）
- rightarm_state_cart_pos: 右臂末端在 torso 坐标系中的位置 [x, y, z, rx, ry, rz]（单位：米）
- torso 坐标系与相机外参共享，无需 torso 全局坐标参与投影计算
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

# ============================================================
# 相机内参（来自 camera_calibration.txt，1408x1280 分辨率，鱼眼模型）
# ============================================================
ORIG_W, ORIG_H = 1408, 1280
VIDEO_W, VIDEO_H = 640, 480

# 原始标定参数
FX_ORIG = 517.1607943040
FY_ORIG = 517.1787653667
CX_ORIG = 703.7844520502
CY_ORIG = 641.3050626158

# 鱼眼畸变系数 (k1, k2, k3, k4)
K1 = 0.0919417169
K2 = -0.0033091746
K3 = -0.0143766221
K4 = 0.0034217546

# 缩放内参到视频分辨率
SCALE_X = VIDEO_W / ORIG_W
SCALE_Y = VIDEO_H / ORIG_H

FX = FX_ORIG * SCALE_X
FY = FY_ORIG * SCALE_Y
CX = CX_ORIG * SCALE_X
CY = CY_ORIG * SCALE_Y

CAMERA_K = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# ============================================================
# 相机外参：T_torso_from_cam（相机坐标系 → torso坐标系）
# 作用：将相机坐标系的点变换到 torso 坐标系
#       P_torso = R @ P_cam + t
#
# 平移向量 t = [0, -72.14, 144.02] mm 是相机在torso坐标系中的位置：
#   tx=0: 左右对齐
#   ty=-72.14mm: 相机在torso前方约7.2cm
#   tz=144.02mm: 相机在torso上方约14.4cm
#
# 旋转矩阵 R: 绕X轴旋转约112度（俯视操作台，光轴向下）
#
# 注意：投影时需要使用逆变换 T_cam_from_torso
# ============================================================
CAMERA_EXTRINSIC = np.array([
    [1.0,  0.0,         0.0,         0.0  ],
    [0.0, -0.37460659, -0.92718385, -72.14],
    [0.0,  0.92718385, -0.37460659,  144.02],
    [0.0,  0.0,         0.0,         1.0  ]
], dtype=np.float64)


def parse_array(val):
    """解析数组列（可能是字符串、列表或 numpy 数组）"""
    if isinstance(val, np.ndarray):
        return val.astype(np.float64)
    if isinstance(val, (list, tuple)):
        return np.array(val, dtype=np.float64)
    if isinstance(val, str):
        val = val.strip().strip('[]')
        return np.array([float(x) for x in val.split()], dtype=np.float64)
    return np.array(val, dtype=np.float64)


def project_fisheye(point_torso, extrinsic, fx, fy, cx, cy, k1, k2, k3, k4):
    """
    将 torso 坐标系中的3D点（单位：米）投影到640x480图像上。

    外参矩阵 extrinsic = T_torso_from_cam:
      将相机坐标系的点变换到 torso 坐标系
      P_torso = R @ P_cam + t

    投影需要逆变换 T_cam_from_torso:
      P_cam = R.T @ (P_torso - t)

    步骤：
    1. 用逆变换将点从 torso 坐标系变换到相机坐标系
    2. 使用鱼眼投影模型投影到图像平面
    3. 输出图像像素坐标

    Returns:
        (u, v) 或 None（点在相机后方时）
    """
    # 提取旋转和平移（平移毫米→米）
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3] / 1000.0

    # torso坐标 -> 相机坐标: P_cam = R.T @ (P_torso - t)
    pt_torso = np.array([point_torso[0], point_torso[1], point_torso[2]])
    pt_cam = R.T @ (pt_torso - t)

    Xc, Yc, Zc = pt_cam[0], pt_cam[1], pt_cam[2]

    if Zc <= 0:
        return None

    # 鱼眼投影
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

    # 当前数据定义下，图像水平/垂直方向均与投影方向相反，做双轴翻转
    u = VIDEO_W - u
    v = VIDEO_H - v

    return (int(round(u)), int(round(v)))


def visualize_task(task_dir, output_dir=None, sample_every=5):
    """
    对整个task目录进行夹爪投影可视化。

    Args:
        task_dir: task目录路径（包含 data/chunk-XXX/ 和 videos/chunk-XXX/）
        output_dir: 输出目录，默认在task目录下创建 projection_vis/
        sample_every: 每隔N帧处理一帧
    """
    task_path = Path(task_dir)
    if output_dir is None:
        output_dir = task_path / "projection_vis"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"任务目录: {task_path}")
    print(f"输出目录: {output_path}")
    print(f"相机内参 (640x480): fx={FX:.2f}, fy={FY:.2f}, cx={CX:.2f}, cy={CY:.2f}")

    # 找所有 chunk 目录下的 parquet 文件
    data_dir = task_path / "data"
    if not data_dir.exists():
        print(f"错误: 找不到数据目录 {data_dir}")
        return

    # 按 episode 组织数据
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"错误: 在 {data_dir} 中未找到 parquet 文件")
        return

    print(f"找到 {len(parquet_files)} 个 parquet 文件")

    # 找视频文件（只使用 cam_high 摄像头）
    video_dir = task_path / "videos"
    video_files = {}
    if video_dir.exists():
        for vf in sorted(video_dir.rglob("*.mp4")):
            # 只使用 cam_high 下的视频
            if "cam_high" not in vf.parts:
                continue
            ep_name = vf.stem  # e.g. episode_000000
            video_files[ep_name] = vf

    print(f"找到 {len(video_files)} 个视频文件")

    total_processed = 0
    total_frames = 0

    for pf in parquet_files:
        ep_name = pf.stem  # e.g. episode_000000
        print(f"\n处理 {ep_name}...")

        # 读取数据
        df = pd.read_parquet(pf)
        print(f"  数据行数: {len(df)}")

        # 检查必要的列
        if 'leftarm_state_cart_pos' not in df.columns:
            print(f"  警告: 找不到 leftarm_state_cart_pos 列，跳过")
            continue
        if 'rightarm_state_cart_pos' not in df.columns:
            print(f"  警告: 找不到 rightarm_state_cart_pos 列，跳过")
            continue

        # 创建对应的输出子目录
        ep_out = output_path / ep_name
        ep_out.mkdir(parents=True, exist_ok=True)

        # 查找对应的视频
        video_path = video_files.get(ep_name)
        if video_path is None:
            print(f"  警告: 找不到对应视频 {ep_name}，跳过")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  警告: 无法打开视频 {video_path}")
            continue

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  视频: {total_video_frames} 帧, {video_w}x{video_h}")

        n_frames = min(len(df), total_video_frames)
        ep_processed = 0

        for frame_idx in range(0, n_frames, sample_every):
            # 读取视频帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            row = df.iloc[frame_idx]

            # 解析左右臂坐标
            try:
                left_pos = parse_array(row['leftarm_state_cart_pos'])[:3]
                right_pos = parse_array(row['rightarm_state_cart_pos'])[:3]
            except Exception as e:
                print(f"  帧{frame_idx} 解析坐标失败: {e}")
                continue

            # 投影左臂
            left_px = project_fisheye(
                left_pos, CAMERA_EXTRINSIC,
                FX, FY, CX, CY, K1, K2, K3, K4
            )
            # 投影右臂
            right_px = project_fisheye(
                right_pos, CAMERA_EXTRINSIC,
                FX, FY, CX, CY, K1, K2, K3, K4
            )

            # 绘制投影点
            vis = frame.copy()
            h, w = vis.shape[:2]

            def draw_point(img, px, label, color_in, color_out):
                if px is None:
                    return img
                u, v = px
                in_frame = 0 <= u < w and 0 <= v < h
                color = color_in if in_frame else color_out
                cv2.circle(img, (u, v), 10, color, -1)
                cv2.circle(img, (u, v), 12, (255, 255, 255), 2)
                cv2.putText(img, label, (u + 14, v + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                cv2.putText(img, label, (u + 14, v + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
                status = "IN" if in_frame else "OUT"
                cv2.putText(img, status, (u + 14, v + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                return img

            vis = draw_point(vis, left_px, 'L',
                             (0, 255, 0), (0, 0, 255))   # 绿=在画面内，红=超出
            vis = draw_point(vis, right_px, 'R',
                             (255, 128, 0), (0, 0, 255))  # 橙=在画面内，红=超出

            # 添加信息文字
            info = f"Frame:{frame_idx}"
            cv2.putText(vis, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if left_px:
                cv2.putText(vis, f"L:({left_px[0]},{left_px[1]})",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "L: behind camera",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

            if right_px:
                cv2.putText(vis, f"R:({right_px[0]},{right_px[1]})",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
            else:
                cv2.putText(vis, "R: behind camera",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

            # 保存
            out_file = ep_out / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_file), vis)
            ep_processed += 1
            total_frames += 1

        cap.release()
        total_processed += ep_processed
        print(f"  完成，保存 {ep_processed} 帧到 {ep_out}")

    print(f"\n全部完成！共处理 {total_frames} 帧，保存到 {output_path}")
    return total_frames


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='夹爪投影可视化')
    parser.add_argument('task_dir', help='task目录路径')
    parser.add_argument('--output', '-o', default=None,
                        help='输出目录（默认: task_dir/projection_vis/）')
    parser.add_argument('--every', '-n', type=int, default=5,
                        help='采样间隔（每N帧处理一帧，默认5）')

    args = parser.parse_args()
    visualize_task(args.task_dir, args.output, args.every)
