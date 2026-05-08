#!/usr/bin/env python3
"""
3D可视化：相机位姿 + 左右夹爪EEF轨迹 + 视锥（修正版）
正确理解外参：T_torso_from_cam（相机坐标系 → torso坐标系）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# ============================================================
# 相机内参（1408x1280，鱼眼模型）
# ============================================================
FX = 517.1607943040
FY = 517.1787653667
CX = 703.7844520502
CY = 641.3050626158
IMG_W = 1408
IMG_H = 1280

# ============================================================
# 相机外参：T_torso_from_cam（相机坐标系 → torso坐标系）
# 公式：P_torso = R @ P_cam + t
#
# 平移向量 t = [0, -72.14, 144.02] mm 是相机原点在torso坐标系中的位置
# 旋转矩阵 R：绕X轴旋转约112度（俯视操作台）
# ============================================================
T_TORSO_FROM_CAM = np.array([
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


def get_camera_pose_in_torso():
    """
    从 T_torso_from_cam 获取相机在torso坐标系中的位姿

    T_torso_from_cam: P_torso = R @ P_cam + t
    - 当 P_cam = 0（相机原点）时，P_torso = t
    - 所以 t 直接就是相机在torso系中的位置

    相机各轴在torso系中的方向：
    - 相机X轴在torso系中 = R @ [1,0,0] = R的第1列
    - 相机Y轴在torso系中 = R @ [0,1,0] = R的第2列
    - 相机Z轴在torso系中 = R @ [0,0,1] = R的第3列（光轴方向）
    """
    R = T_TORSO_FROM_CAM[:3, :3]
    t_mm = T_TORSO_FROM_CAM[:3, 3]
    t_m = t_mm / 1000.0  # mm -> m

    # 相机位置（torso系）
    cam_pos_torso = t_m

    # 相机各轴在torso系中的方向（R的列向量）
    cam_x_torso = R[:, 0]  # R的第1列
    cam_y_torso = R[:, 1]  # R的第2列
    cam_z_torso = R[:, 2]  # R的第3列（光轴）

    return cam_pos_torso, cam_z_torso, cam_x_torso, cam_y_torso, R, t_m


def get_frustum_corners(cam_pos, R, t, depth=0.3):
    """
    计算视锥在torso坐标系中的角点

    步骤：
    1. 图像4个角点 -> 相机坐标系中的3D方向（鱼眼反投影）
    2. 使用 T_torso_from_cam 将这些点变换到torso坐标系
    """
    corners_img = [
        [0, 0],
        [IMG_W, 0],
        [IMG_W, IMG_H],
        [0, IMG_H]
    ]

    # 鱼眼反投影：像素 -> 相机坐标系中的3D方向
    corners_cam = []
    for u, v in corners_img:
        xd = (u - CX) / FX
        yd = (v - CY) / FY
        rd = np.sqrt(xd**2 + yd**2)

        if rd < 1e-6:
            d = np.array([0, 0, depth])
        else:
            theta = rd  # 简化：theta ≈ theta_d
            r = np.tan(theta)
            x = r * (xd / rd)
            y = r * (yd / rd)
            z = 1.0
            d = np.array([x, y, z])
            d = d / np.linalg.norm(d) * depth

        corners_cam.append(d)

    # 变换到torso坐标系：P_torso = R @ P_cam + t
    corners_torso = []
    for c in corners_cam:
        p_torso = R @ c + cam_pos
        corners_torso.append(p_torso)

    return cam_pos, corners_torso


def draw_frustum(ax, cam_pos, corners, color='cyan', alpha=0.1):
    """绘制视锥"""
    for i in range(4):
        j = (i + 1) % 4
        verts = [[cam_pos, corners[i], corners[j]]]
        poly = Poly3DCollection(verts, alpha=alpha, facecolor=color,
                                edgecolor=color, linewidth=0.5)
        ax.add_collection3d(poly)

    verts = [[corners[0], corners[1], corners[2], corners[3]]]
    poly = Poly3DCollection(verts, alpha=alpha*0.5, facecolor=color,
                            edgecolor=color, linewidth=0.5)
    ax.add_collection3d(poly)

    for c in corners:
        ax.plot3D(*zip(cam_pos, c), color=color, alpha=0.4, linewidth=0.8)


def euler_to_rotation_matrix(rx, ry, rz):
    """欧拉角转旋转矩阵"""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def main():
    parquet_file = '/mnt/vepfs01/output/klayzhou/quality_check/Gripperout_data/task_2689/data/chunk-000/episode_000000.parquet'

    print("读取数据...")
    df = pd.read_parquet(parquet_file)
    print(f"总行数: {len(df)}")

    # 采样夹爪位置
    step = 50
    left_positions = []
    left_directions = []
    right_positions = []
    right_directions = []

    for i in range(0, len(df), step):
        row = df.iloc[i]
        left = parse_array(row['leftarm_cmd_cart_pos'])
        right = parse_array(row['rightarm_cmd_cart_pos'])

        if len(left) >= 3:
            left_positions.append(left[:3])
        if len(left) >= 6:
            R_left = euler_to_rotation_matrix(left[3], left[4], left[5])
            left_directions.append(R_left[:, 2])

        if len(right) >= 3:
            right_positions.append(right[:3])
        if len(right) >= 6:
            R_right = euler_to_rotation_matrix(right[3], right[4], right[5])
            right_directions.append(R_right[:, 2])

    left_positions = np.array(left_positions)
    right_positions = np.array(right_positions)
    left_directions = np.array(left_directions) if left_directions else None
    right_directions = np.array(right_directions) if right_directions else None

    print(f"左臂采样点: {len(left_positions)}")
    print(f"右臂采样点: {len(right_positions)}")

    # 获取相机位姿
    cam_pos, cam_z, cam_x, cam_y, R, t = get_camera_pose_in_torso()
    cam_frustum_pos, frustum_corners = get_frustum_corners(cam_pos, R, t, depth=0.3)

    print(f"\n相机在torso系中的位置: {cam_pos}")
    print(f"相机Z轴方向（光轴）: {cam_z}")
    print(f"相机X轴方向: {cam_x}")
    print(f"相机Y轴方向: {cam_y}")

    # ============================================================
    # 绘图
    # ============================================================
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 相机位置
    ax.scatter(*cam_pos, color='cyan', s=200, zorder=10, label='Camera')
    ax.text(cam_pos[0]+0.02, cam_pos[1]+0.02, cam_pos[2]+0.02,
            'Camera', fontsize=10, color='cyan', fontweight='bold')

    # 相机坐标轴
    axis_len = 0.08
    ax.quiver(*cam_pos, *cam_x, length=axis_len, color='red',
              linewidth=2, arrow_length_ratio=0.3)
    ax.quiver(*cam_pos, *cam_y, length=axis_len, color='green',
              linewidth=2, arrow_length_ratio=0.3)
    ax.quiver(*cam_pos, *cam_z, length=axis_len*2, color='blue',
              linewidth=3, arrow_length_ratio=0.2, label='Camera Z (optical axis)')

    # 视锥
    draw_frustum(ax, cam_pos, frustum_corners, color='cyan', alpha=0.12)

    # Torso原点
    ax.scatter(0, 0, 0, color='black', s=300, marker='*', zorder=10, label='Torso origin')
    ax.text(0.02, 0.02, 0.02, 'Torso', fontsize=10, color='black', fontweight='bold')

    # Torso坐标轴
    torso_len = 0.15
    ax.quiver(0, 0, 0, torso_len, 0, 0, color='red', linewidth=1.5,
              arrow_length_ratio=0.3, alpha=0.5)
    ax.quiver(0, 0, 0, 0, torso_len, 0, color='green', linewidth=1.5,
              arrow_length_ratio=0.3, alpha=0.5)
    ax.quiver(0, 0, 0, 0, 0, torso_len, color='blue', linewidth=1.5,
              arrow_length_ratio=0.3, alpha=0.5)

    # 左臂轨迹
    ax.plot(left_positions[:, 0], left_positions[:, 1], left_positions[:, 2],
            'g-', alpha=0.4, linewidth=1.5, label='Left arm')
    ax.scatter(left_positions[:, 0], left_positions[:, 1], left_positions[:, 2],
               c='lime', s=15, alpha=0.6, zorder=5)

    if left_directions is not None:
        arrow_step = 5
        dir_len = 0.04
        for i in range(0, len(left_positions), arrow_step):
            if i < len(left_directions):
                ax.quiver(left_positions[i, 0], left_positions[i, 1], left_positions[i, 2],
                          left_directions[i, 0], left_directions[i, 1], left_directions[i, 2],
                          length=dir_len, color='darkgreen', alpha=0.6,
                          arrow_length_ratio=0.4, linewidth=0.8)

    ax.scatter(*left_positions[0], color='green', s=100, marker='D', zorder=10)
    ax.text(*left_positions[0], ' L-start', fontsize=8, color='green')
    ax.scatter(*left_positions[-1], color='darkgreen', s=100, marker='s', zorder=10)
    ax.text(*left_positions[-1], ' L-end', fontsize=8, color='darkgreen')

    # 右臂轨迹
    ax.plot(right_positions[:, 0], right_positions[:, 1], right_positions[:, 2],
            'r-', alpha=0.4, linewidth=1.5, label='Right arm')
    ax.scatter(right_positions[:, 0], right_positions[:, 1], right_positions[:, 2],
               c='orange', s=15, alpha=0.6, zorder=5)

    if right_directions is not None:
        for i in range(0, len(right_positions), arrow_step):
            if i < len(right_directions):
                ax.quiver(right_positions[i, 0], right_positions[i, 1], right_positions[i, 2],
                          right_directions[i, 0], right_directions[i, 1], right_directions[i, 2],
                          length=dir_len, color='darkorange', alpha=0.6,
                          arrow_length_ratio=0.4, linewidth=0.8)

    ax.scatter(*right_positions[0], color='red', s=100, marker='D', zorder=10)
    ax.text(*right_positions[0], ' R-start', fontsize=8, color='red')
    ax.scatter(*right_positions[-1], color='darkred', s=100, marker='s', zorder=10)
    ax.text(*right_positions[-1], ' R-end', fontsize=8, color='darkred')

    # 图形设置
    ax.set_xlabel('X (m) - LEFT', fontsize=11)  # X轴向左
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)

    # 反转X轴，因为torso坐标系X轴向左（正X=左，负X=右）
    ax.invert_xaxis()

    ax.set_title(f'Gripper EEF Trajectory + Camera Pose + Frustum (FIXED)\n'
                 f'Torso frame: X-left, Y-forward, Z-up\n'
                 f'Camera pos: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}) m\n'
                 f'Camera Z-axis: ({cam_z[0]:.3f}, {cam_z[1]:.3f}, {cam_z[2]:.3f})',
                 fontsize=12)

    legend_elements = [
        mpatches.Patch(color='cyan', alpha=0.7, label='Camera + Frustum'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Left arm'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Right arm'),
        plt.Line2D([0], [0], marker='*', color='black', markersize=12,
                   linestyle='None', label='Torso origin'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Camera Z-axis (optical)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    info_text = (
        f'Camera extrinsic: T_torso_from_cam\n'
        f'  Translation: ({T_TORSO_FROM_CAM[0,3]:.1f}, {T_TORSO_FROM_CAM[1,3]:.1f}, '
        f'{T_TORSO_FROM_CAM[2,3]:.1f}) mm\n'
        f'  Rotation: ~22deg around X-axis\n'
        f'Camera position (torso frame):\n'
        f'  ({cam_pos[0]*1000:.1f}, {cam_pos[1]*1000:.1f}, {cam_pos[2]*1000:.1f}) mm\n'
        f'Camera Z-axis (optical axis):\n'
        f'  ({cam_z[0]:.3f}, {cam_z[1]:.3f}, {cam_z[2]:.3f})'
    )
    fig.text(0.02, 0.02, info_text, fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = '/mnt/vepfs01/output/klayzhou/quality_check/gripper_3d_visualization_fixed.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 3D可视化图像已保存到: {output_path}")

    print(f"\n=== 关键信息 ===")
    print(f"相机位置 (torso系, m): {cam_pos}")
    print(f"相机Z轴方向 (光轴): {cam_z}")
    print(f"  -> Z分量={cam_z[2]:.3f} ({'向上' if cam_z[2]>0 else '向下'})")


if __name__ == '__main__':
    main()
