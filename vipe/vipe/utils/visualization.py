# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont
from pycg import image

from vipe.ext.lietorch import SE3
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import CachedVideoStream, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional

from .geometry import project_points_to_panorama, project_points_to_pinhole


rng = np.random.RandomState(200)
_palette = ((rng.random((3 * 255)) * 0.7 + 0.3) * 255).astype(np.uint8).tolist()
_palette = [0, 0, 0] + _palette

POINTS_STENCIL = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
POINTS_STENCIL = np.stack(POINTS_STENCIL, axis=-1).reshape(-1, 2)
POINTS_STENCIL = POINTS_STENCIL[np.max(np.abs(POINTS_STENCIL), axis=-1) > 1]
POINTS_STENCIL = np.pad(POINTS_STENCIL, ((0, 1), (0, 0)), constant_values=0)


class VideoWriter:
    """
    Simple video writer class (use h264 codec).

    Usage:
    ```
    with VideoWriter("output.mp4", 30) as vw:
        for frame in frames:
            vw.write(frame)
    ```
    """

    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps
        self.vw: Any = None

    def __enter__(self):
        return self

    def write(self, frame: np.ndarray):
        if self.vw is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.vw = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", macro_block_size=None)

        if frame.dtype in [np.float32, np.float64]:
            frame = (frame * 255).astype(np.uint8)

        assert self.vw is not None
        self.vw.append_data(frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.vw is not None:
            self.vw.close()


def bbox_with_size(pcd_xyz: torch.Tensor, quantile: float = 0.98):
    from pycg import vis

    low_quantile, high_quantile = (1 - quantile) / 2, 1 - (1 - quantile) / 2
    pcd_min = torch.quantile(pcd_xyz, low_quantile, dim=0, keepdim=True)
    pcd_max = torch.quantile(pcd_xyz, high_quantile, dim=0, keepdim=True)

    x_length = pcd_max[0, 0] - pcd_min[0, 0]
    x_length_pos = pcd_min[0] + torch.tensor([x_length / 2, 0, 0])
    y_length = pcd_max[0, 1] - pcd_min[0, 1]
    y_length_pos = pcd_min[0] + torch.tensor([0, y_length / 2, 0])
    z_length = pcd_max[0, 2] - pcd_min[0, 2]
    z_length_pos = pcd_min[0] + torch.tensor([0, 0, z_length / 2])

    return [
        vis.wireframe_bbox(pcd_min, pcd_max, ucid=-1),
        vis.text(f"{x_length.item():.2f}m", x_length_pos),
        vis.text(f"{y_length.item():.2f}m", y_length_pos),
        vis.text(f"{z_length.item():.2f}m", z_length_pos),
    ]


def colorize_mask(pred_mask: np.ndarray):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode="P")
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode="RGB")
    return np.array(save_mask)


def colorize_depth(
    depth: np.ndarray,
    normalize: bool = False,
    clip_depth: bool = False,
    min_depth: float = 1e-3,
    max_depth: float = 1e4,
):
    if clip_depth:
        depth = np.clip(depth, a_min=min_depth, a_max=max_depth)

    if normalize:
        depth = (depth - depth.min()) / (depth.max() - depth.min())

    depth = (depth * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth


def splatting_render(
    xyz: np.ndarray,
    pose: SE3,
    intrinsics: np.ndarray,
    camera_type: CameraType,
    frame_size: tuple[int, int],
    subsample_factor: int,
    color: np.ndarray | None = None,
    is_panorama: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    使用splatting技术渲染点云 (向量化版本，借鉴您提供代码的优化)
    返回: (彩色图像, 掩码图像)
    """
    assert pose.shape == (), "Only single pose is supported"
    
    target_h, target_w = frame_size[0], frame_size[1]
    
    # 变换到相机坐标系
    pose_matrix = pose.inv().matrix().cpu()
    local_xyz = torch.from_numpy(xyz) @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]
    
    # 投影到图像平面
    if is_panorama:
        uv = project_points_to_panorama(local_xyz, return_depth=False)
        uv[:, 0] *= target_w
        uv[:, 1] *= target_h
        depths = torch.norm(local_xyz, dim=1)  # 对于全景图使用距离
    else:
        camera_model = camera_type.build_camera_model(torch.from_numpy(intrinsics)).scaled(1.0 / subsample_factor)
        xyz_h = torch.cat([local_xyz, torch.ones((local_xyz.shape[0], 1))], dim=1)
        uv, _, _ = camera_model.proj_points(xyz_h)
        depths = local_xyz[:, 2]  # z depth
        
        # 过滤掉相机后面的点和超出边界的点 - 增加更严格的深度过滤
        valid = (depths > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < target_w) & (uv[:, 1] >= 0) & (uv[:, 1] < target_h)
        uv = uv[valid]
        depths = depths[valid]
        local_xyz = local_xyz[valid]
        if color is not None:
            color = color[valid]
    
    if uv.shape[0] == 0:  # 没有有效点
        canvas_color = torch.zeros((target_h, target_w, 3), dtype=torch.uint8)
        canvas_mask = torch.zeros((target_h, target_w), dtype=torch.uint8)
        return canvas_color.numpy(), canvas_mask.numpy()
    
    # 准备颜色数据 - 改进颜色处理
    if color is not None:
        if isinstance(color, np.ndarray):
            point_colors = torch.from_numpy(color).float()
            # 确保颜色值在合理范围内
            if point_colors.max() > 1.0:
                point_colors = point_colors / 255.0
            # 将[0,1]范围的颜色保持不变，不转换到[-1,1]
            point_colors = torch.clamp(point_colors, 0.0, 1.0)  # 确保在[0,1]范围
        else:
            point_colors = torch.tensor(color).float().expand(uv.shape[0], 3)
            if point_colors.max() > 1.0:
                point_colors = point_colors / 255.0
            point_colors = torch.clamp(point_colors, 0.0, 1.0)
    else:
        point_colors = torch.ones((uv.shape[0], 3), dtype=torch.float32)  # 默认白色
    
    # Splatting: 每个点影响2x2像素区域
    u = uv[:, 0]
    v = uv[:, 1]
    
    u_floor = torch.floor(u).long()
    v_floor = torch.floor(v).long()
    u_ceil = torch.clamp(u_floor + 1, max=target_w - 1)
    v_ceil = torch.clamp(v_floor + 1, max=target_h - 1)
    
    # 双线性插值权重
    wu = u - u_floor.float()
    wv = v - v_floor.float()
    
    w_tl = (1 - wu) * (1 - wv)  # 左上角
    w_tr = wu * (1 - wv)        # 右上角
    w_bl = (1 - wu) * wv        # 左下角
    w_br = wu * wv              # 右下角
    
    # 深度权重 (closer points have higher weight) - 改进深度权重计算
    depth_weights = 1.0 / (torch.abs(depths) + 1e-6)
    
    # 初始化输出buffer - 使用float32精度避免累积误差
    rendered_image = torch.zeros((3, target_h, target_w), dtype=torch.float32)
    weight_buffer = torch.zeros((target_h, target_w), dtype=torch.float32)
    
    # 借鉴您代码的向量化scatter操作
    # 计算所有四个角的线性索引
    idx_tl = v_floor * target_w + u_floor
    idx_tr = v_floor * target_w + u_ceil  
    idx_bl = v_ceil * target_w + u_floor
    idx_br = v_ceil * target_w + u_ceil
    
    # 计算所有权重
    weight_tl = w_tl * depth_weights
    weight_tr = w_tr * depth_weights
    weight_bl = w_bl * depth_weights
    weight_br = w_br * depth_weights
    
    # 向量化散布 - 使用 scatter_add_
    flat_image = rendered_image.view(3, -1)  # (3, H*W)
    flat_weight = weight_buffer.view(-1)     # (H*W,)
    
    # 边界检查 - 确保索引有效
    valid_tl = (v_floor >= 0) & (v_floor < target_h) & (u_floor >= 0) & (u_floor < target_w)
    valid_tr = (v_floor >= 0) & (v_floor < target_h) & (u_ceil >= 0) & (u_ceil < target_w)
    valid_bl = (v_ceil >= 0) & (v_ceil < target_h) & (u_floor >= 0) & (u_floor < target_w)
    valid_br = (v_ceil >= 0) & (v_ceil < target_h) & (u_ceil >= 0) & (u_ceil < target_w)
    
    # Top-left
    if valid_tl.any():
        idx_tl_valid = idx_tl[valid_tl]
        weight_tl_valid = weight_tl[valid_tl]
        color_tl_valid = point_colors[valid_tl]
        flat_image.scatter_add_(1, idx_tl_valid.unsqueeze(0).expand(3, -1), (color_tl_valid * weight_tl_valid.unsqueeze(1)).T)
        flat_weight.scatter_add_(0, idx_tl_valid, weight_tl_valid)
    
    # Top-right
    if valid_tr.any():
        idx_tr_valid = idx_tr[valid_tr]
        weight_tr_valid = weight_tr[valid_tr]
        color_tr_valid = point_colors[valid_tr]
        flat_image.scatter_add_(1, idx_tr_valid.unsqueeze(0).expand(3, -1), (color_tr_valid * weight_tr_valid.unsqueeze(1)).T)
        flat_weight.scatter_add_(0, idx_tr_valid, weight_tr_valid)
    
    # Bottom-left
    if valid_bl.any():
        idx_bl_valid = idx_bl[valid_bl]
        weight_bl_valid = weight_bl[valid_bl]
        color_bl_valid = point_colors[valid_bl]
        flat_image.scatter_add_(1, idx_bl_valid.unsqueeze(0).expand(3, -1), (color_bl_valid * weight_bl_valid.unsqueeze(1)).T)
        flat_weight.scatter_add_(0, idx_bl_valid, weight_bl_valid)
    
    # Bottom-right
    if valid_br.any():
        idx_br_valid = idx_br[valid_br]
        weight_br_valid = weight_br[valid_br]
        color_br_valid = point_colors[valid_br]
        flat_image.scatter_add_(1, idx_br_valid.unsqueeze(0).expand(3, -1), (color_br_valid * weight_br_valid.unsqueeze(1)).T)
        flat_weight.scatter_add_(0, idx_br_valid, weight_br_valid)
    
    # Reshape back
    rendered_image = flat_image.view(3, target_h, target_w)
    weight_buffer = flat_weight.view(target_h, target_w)
    
    # 归一化 - 改进归一化过程
    valid_pixels = weight_buffer > 1e-6  # 使用更严格的阈值
    
    # 避免除零，只对有效像素进行归一化
    rendered_image[:, valid_pixels] = rendered_image[:, valid_pixels] / weight_buffer[valid_pixels].unsqueeze(0)
    
    # 将HWC格式转换为与原代码兼容的格式
    rendered_image = rendered_image.permute(1, 2, 0)  # (H, W, 3)
    
    # 转换回numpy并调整范围 - 改进数值稳定性
    canvas_color_np = torch.clamp(rendered_image * 255, 0, 255).byte().numpy()
    canvas_mask_np = (valid_pixels * 255).byte().numpy()
    
    
    return canvas_color_np, canvas_mask_np


def draw_points_batch(
    canvas: np.ndarray,
    pts: np.ndarray,
    color: np.ndarray | tuple | None = None,
    stencil: np.ndarray | None = None,
):
    if stencil is None:
        stencil = np.array([[0, 0]])

    for rel_pos in stencil:
        pos = (pts + rel_pos[None]).astype(int)
        in_bound = (pos[:, 0] >= 0) & (pos[:, 0] < canvas.shape[1]) & (pos[:, 1] >= 0) & (pos[:, 1] < canvas.shape[0])
        pos = pos[in_bound]
        if isinstance(color, np.ndarray):
            canvas[pos[:, 1], pos[:, 0]] = color[in_bound]
        else:
            canvas[pos[:, 1], pos[:, 0]] = color or (0, 255, 0)
    return canvas


def draw_lines_batch(
    canvas: np.ndarray,
    lines_start: np.ndarray,
    lines_end: np.ndarray,
    color: tuple | None = None,
):
    if lines_start.shape[0] == 0:
        return canvas
    lines = np.stack([lines_start, lines_end], axis=1).astype(int)
    return cv2.polylines(
        canvas.copy(),
        [l for l in lines],
        isClosed=False,
        color=color or (0, 255, 0),
        thickness=1,
    )


def draw_tracks(canvas: np.ndarray, tracks: np.ndarray, valid: np.ndarray):
    """
    Args:
        canvas: The image to draw the tracks on. (H, W, 3) uint8
        tracks: The tracks to draw. (length, n_tracks, 2)
            To draw tracks of different lengths, please call this function multiple times.
        valid: The validity of the tracks. (length, n_tracks)
    """
    for l in range(tracks.shape[0]):
        uv, uv_valid = tracks[l], valid[l]
        canvas = draw_points_batch(canvas, uv[uv_valid], (0, 255 - 20 * l, 0), stencil=POINTS_STENCIL)
    for l in range(tracks.shape[0] - 1):
        uv_start, start_valid = tracks[l], valid[l]
        uv_end, end_valid = tracks[l + 1], valid[l + 1]
        all_valid = start_valid & end_valid
        canvas = draw_lines_batch(canvas, uv_start[all_valid], uv_end[all_valid], (0, 255 - 20 * l, 0))
    return canvas


def project_points_panorama(
    xyz: np.ndarray,
    pose: SE3,
    frame_size: tuple[int, int],
    color: np.ndarray | None = None,
) -> np.ndarray:
    assert pose.shape == (), "Only single pose is supported"

    canvas = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255

    pose_matrix = pose.inv().matrix().cpu().numpy()
    local_xyz = xyz @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]

    uv = project_points_to_panorama(torch.from_numpy(local_xyz), return_depth=False)
    uv[:, 0] *= frame_size[1]
    uv[:, 1] *= frame_size[0]
    uv = (uv - 0.5).round().int().cpu().numpy()

    if color is not None:
        if np.issubdtype(color.dtype, np.floating):
            color = (color * 255).astype(np.uint8)

    return draw_points_batch(canvas, uv, color, stencil=POINTS_STENCIL)


def project_points(
    xyz: np.ndarray,
    intrinsics: np.ndarray,
    camera_type: CameraType,
    pose: SE3,
    frame_size: tuple[int, int],
    subsample_factor: int,
    color: np.ndarray | None = None,
) -> np.ndarray:
    assert pose.shape == (), "Only single pose is supported"

    canvas = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255

    pose_matrix = pose.inv().matrix().cpu().numpy()
    local_xyz = xyz @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]

    camera_model = camera_type.build_camera_model(torch.from_numpy(intrinsics)).scaled(1.0 / subsample_factor)
    xyz_h = torch.cat([torch.from_numpy(local_xyz), torch.ones((local_xyz.shape[0], 1))], dim=1)
    uv, _, _ = camera_model.proj_points(xyz_h)
    in_bound = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < frame_size[1])
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < frame_size[0])
        & torch.from_numpy(local_xyz[:, 2] > 0)
    )
    uv = uv[in_bound]
    uv = (uv - 0.5).round().int().cpu().numpy()

    # uv, in_bound = project_points_to_pinhole(
    #     torch.from_numpy(local_xyz),
    #     torch.from_numpy(intrinsics),
    #     frame_size,
    #     return_depth=False,
    # )
    # uv = uv[in_bound]
    # uv[:, 0] *= frame_size[1]
    # uv[:, 1] *= frame_size[0]
    # uv = (uv - 0.5).round().int().cpu().numpy()

    if color is not None:
        color = color[in_bound]
        if np.issubdtype(color.dtype, np.floating):
            color = (color * 255).astype(np.uint8)

    return draw_points_batch(canvas, uv, color, stencil=POINTS_STENCIL)


def image_above_text(img: np.ndarray, text: str = "<TEXT>") -> Image.Image:
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    image = Image.fromarray(img)

    width, height = image.size
    text_height = max(20, height // 10)

    new_height = height + int(text_height * 1.5)
    new_image = Image.new("RGB", (width, new_height), color=(255, 255, 255))
    new_image.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_image)

    try:
        font = ImageFont.truetype("arial.ttf", text_height)  # You can change the font size
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if arial is not available

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (width - text_width) // 2
    text_y = height + int(text_height * 0.2)

    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))  # Black text
    return new_image


def save_projection_video(
    video_path: Path,
    video_stream: VideoStream,
    slam_output: SLAMOutput | None,
    subsample_factor: int,
    attributes: list[list[str]],
):
    assert isinstance(video_stream, CachedVideoStream)

    img_h, img_w = video_stream.frame_size()
    img_h //= subsample_factor
    img_w //= subsample_factor

    na_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    text_img = image.text("N/A")
    na_img = image.place_image(
        text_img,
        na_img,
        img_w // 2 - text_img.shape[1] // 2,
        img_h // 2 - text_img.shape[0] // 2,
    )
    na_img = (na_img[..., :3] * 255).astype(np.uint8)

    def get_depth_imgs():
        depth_range = [np.inf, -np.inf]

        # Run first to obtain depth range
        for frame_data in video_stream:
            assert isinstance(frame_data, VideoFrame)

            if (depth_data := frame_data.metric_depth) is None:
                continue
            depth_data = depth_data.reciprocal()

            # Remove sky regions if any
            depth_data = depth_data[~frame_data.sky_mask & torch.isfinite(depth_data)]

            depth_min_q, depth_max_q = torch.quantile(depth_data, torch.tensor([0.05, 0.95], device=depth_data.device))
            depth_range[0] = min(depth_range[0], depth_min_q.item())
            depth_range[1] = max(depth_range[1], depth_max_q.item())
        depth_middle = (depth_range[0] + depth_range[1]) / 2
        depth_scale = depth_range[1] - depth_range[0]
        depth_min = depth_middle - depth_scale / 2 * 1.3
        depth_max = depth_middle + depth_scale / 2 * 1.3

        # Then output normalized depth
        for frame_data in video_stream:
            if (depth_data := frame_data.metric_depth) is None:
                yield na_img
                continue

            depth_data = depth_data.reciprocal()
            depth_data[frame_data.sky_mask] = depth_min
            depth_data[~torch.isfinite(depth_data)] = depth_min

            depth_data = depth_data[::subsample_factor, ::subsample_factor]
            depth_img = depth_data.cpu().numpy().astype(float)
            depth_img = (depth_img - depth_min) / (depth_max - depth_min)
            depth_img = np.clip(depth_img, 0, 1)
            yield colorize_depth(depth_img)

    def get_pcd_imgs():
        assert slam_output is not None, "SLAM output is required!"
        slam_map = unpack_optional(slam_output.slam_map)
        pcd_xyz = slam_map.dense_disp_xyz.cpu().numpy()
        pcd_rgb = slam_map.dense_disp_rgb.cpu().numpy()
        for frame_data in video_stream:
            assert isinstance(frame_data, VideoFrame)
            intrinsics = unpack_optional(frame_data.intrinsics)
            if torch.sum(intrinsics) < 1e-6:
                pcd_img, mask_img = splatting_render(
                    pcd_xyz,
                    frame_data.pose,
                    frame_data.intrinsics.cpu().numpy(),
                    frame_data.camera_type,
                    frame_size=(img_h, img_w),
                    subsample_factor=subsample_factor,
                    color=pcd_rgb,
                    is_panorama=True,
                )
            else:
                pcd_img, mask_img = splatting_render(
                    pcd_xyz,
                    frame_data.pose,
                    frame_data.intrinsics.cpu().numpy(),
                    frame_data.camera_type,
                    frame_size=(img_h, img_w),
                    subsample_factor=subsample_factor,
                    color=pcd_rgb,
                    is_panorama=False,
                )
            yield pcd_img

    def get_mask_imgs():
        assert slam_output is not None, "SLAM output is required!"
        slam_map = unpack_optional(slam_output.slam_map)
        pcd_xyz = slam_map.dense_disp_xyz.cpu().numpy()
        for frame_data in video_stream:
            assert isinstance(frame_data, VideoFrame)
            intrinsics = unpack_optional(frame_data.intrinsics)
            if torch.sum(intrinsics) < 1e-6:
                _, mask_img = splatting_render(
                    pcd_xyz,
                    frame_data.pose,
                    frame_data.intrinsics.cpu().numpy(),
                    frame_data.camera_type,
                    frame_size=(img_h, img_w),
                    subsample_factor=subsample_factor,
                    color=None,  # 使用默认白色
                    is_panorama=True,
                )
            else:
                _, mask_img = splatting_render(
                    pcd_xyz,
                    frame_data.pose,
                    frame_data.intrinsics.cpu().numpy(),
                    frame_data.camera_type,
                    frame_size=(img_h, img_w),
                    subsample_factor=subsample_factor,
                    color=None,  # 使用默认白色
                    is_panorama=False,
                )
            # 转换为黑白掩码：有点的地方是白色(255)，没有的地方是黑色(0)
            mask_3ch = np.stack([mask_img, mask_img, mask_img], axis=2)
            yield mask_3ch

    def get_rectified_imgs():
        # Obtain rectification map
        for frame_data in video_stream:
            original_intr = frame_data.camera_type.build_camera_model(frame_data.intrinsics).scaled(
                1 / subsample_factor
            )
            pinhole_intr = original_intr.pinhole()
            device = pinhole_intr.intrinsics.device
            y, x = torch.meshgrid(torch.arange(img_h).float(), torch.arange(img_w).float(), indexing="ij")
            y, x = y.to(device), x.to(device)
            pts, _, _ = pinhole_intr.iproj_disp(torch.ones_like(x), x, y)
            coords, _, _ = original_intr.proj_points(pts)
            coords_norm = 2.0 * coords / torch.tensor([img_w, img_h], device=coords.device) - 1.0
            coords_norm = coords_norm.reshape(1, img_h, img_w, 2)
            break
        for frame_data in video_stream:
            assert isinstance(frame_data, VideoFrame)
            img = frame_data.rgb.permute(2, 0, 1).unsqueeze(0)
            img = torch.nn.functional.grid_sample(
                img,
                coords_norm,
                mode="bilinear",
                align_corners=False,
            )[0].float()
            img = img.permute(1, 2, 0).cpu().numpy()
            yield (img * 255).astype(np.uint8)

    def get_rgb_imgs():
        for frame_data in video_stream:
            rgb_img = frame_data.rgb.cpu().numpy().astype(float)
            rgb_img = (rgb_img * 255).astype(np.uint8)
            rgb_img = cv2.resize(rgb_img, (img_w, img_h))
            yield rgb_img

    def get_instance_imgs():
        for frame_data, rgb_img in zip(video_stream, get_rgb_imgs()):
            assert isinstance(frame_data, VideoFrame)
            if frame_data.instance is None:
                yield na_img
                continue
            instance_img = (inst_np := frame_data.instance.cpu().numpy()).astype(float)
            instance_img = colorize_mask(instance_img)

            if frame_data.instance_phrases is not None:
                for instance_id, instance_phrase in frame_data.instance_phrases.items():
                    if instance_id <= 0:
                        continue
                    text_img = image.text(instance_phrase)
                    inst_mask = inst_np == instance_id
                    try:
                        h_min, h_max = np.where(np.any(inst_mask, axis=1))[0][[0, -1]]
                        w_min, w_max = np.where(np.any(inst_mask, axis=0))[0][[0, -1]]
                        instance_img = image.place_image(
                            text_img,
                            instance_img,
                            (w_min + w_max) // 2,
                            (h_min + h_max) // 2,
                        )
                    except IndexError:
                        pass

            if instance_img.dtype == np.float64:
                instance_img = (instance_img[..., :3] * 255).astype(np.uint8)
            instance_img = cv2.resize(instance_img, (img_w, img_h))
            yield cv2.addWeighted(rgb_img, 0.5, instance_img, 0.5, 0)

    def get_empty_imgs():
        for _ in range(len(video_stream)):
            yield na_img

    # 修改这部分：只处理点云投影
    img_iterators = [
        [
            {
                "rgb": get_rgb_imgs(),
                "depth": get_depth_imgs(),
                "pcd": get_pcd_imgs(),
                "instance": get_instance_imgs(),
                "rectified": get_rectified_imgs(),
                "empty": get_empty_imgs(),
            }[t]
            for t in t_arr
        ]
        for t_arr in attributes
    ]
    
    # 生成vis.mp4 - 纯点云投影结果
    vis_video_path = video_path.parent / (video_path.stem + "_pcd" + video_path.suffix)
    mask_video_path = video_path.parent / (video_path.stem + "_mask" + video_path.suffix)
    
    with VideoWriter(vis_video_path, video_stream.fps()) as vw_vis, \
         VideoWriter(mask_video_path, video_stream.fps()) as vw_mask:
        
        for frame_idx, frame_data in pbar(enumerate(video_stream), total=len(video_stream), desc="Writing viz video"):
            # 获取点云投影结果
            assert slam_output is not None, "SLAM output is required!"
            slam_map = unpack_optional(slam_output.slam_map)
            pcd_xyz = slam_map.dense_disp_xyz.cpu().numpy()
            pcd_rgb = slam_map.dense_disp_rgb.cpu().numpy()
            
            intrinsics = unpack_optional(frame_data.intrinsics)
            if torch.sum(intrinsics) < 1e-6:
                pcd_img, mask_img = splatting_render(
                    pcd_xyz,
                    frame_data.pose,
                    frame_data.intrinsics.cpu().numpy(),
                    frame_data.camera_type,
                    frame_size=(img_h, img_w),
                    subsample_factor=subsample_factor,
                    color=pcd_rgb,
                    is_panorama=True,
                )
            else:
                pcd_img, mask_img = splatting_render(
                    pcd_xyz,
                    frame_data.pose,
                    frame_data.intrinsics.cpu().numpy(),
                    frame_data.camera_type,
                    frame_size=(img_h, img_w),
                    subsample_factor=subsample_factor,
                    color=pcd_rgb,
                    is_panorama=False,
                )
            
            
            # Resize到目标尺寸 (512, 1024)
            target_size = (1024, 512)  # OpenCV的resize函数参数是 (width, height)
            pcd_img_resized = cv2.resize(pcd_img, target_size)

            # 为mask.mp4生成黑白掩码并resize
            mask_3ch = np.stack([mask_img, mask_img, mask_img], axis=2)
            mask_3ch_resized = cv2.resize(mask_3ch, target_size)

            # 写入resize后的结果
            vw_vis.write(pcd_img_resized)
            vw_mask.write(mask_3ch_resized)