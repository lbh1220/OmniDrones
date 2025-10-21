import torch
import torch.nn.functional as F
import math
import os


def extend_occupancy_map(
    occ_grid: torch.Tensor,
    robot_radius: float,
    grid_size: float
) -> torch.Tensor:
    """
    使用高效的形态学膨胀操作（通过最大池化实现）来扩展地图中的障碍物。
    这是一个独立的工具函数。

    Args:
        occ_grid: 原始的布尔类型占据栅格，形状应为 [H, W]。
        robot_radius: 机器人的半径，单位为米。
        grid_size: 栅格地图的分辨率，即每个像素代表的物理尺寸（米）。

    Returns:
        一个新的布尔类型占据栅格，形状为 [H, W]，其中障碍物被扩展了指定的半径。
    """
    # -- 1. 检查输入形状 --
    if occ_grid.dim() != 2:
        raise ValueError(f"Input occ_grid must be a 2D tensor of shape [H, W], but got shape {occ_grid.shape}")

    # -- 2. 计算膨胀核（Kernel）的大小 --
    # 将世界单位的半径转换为像素单位
    # 我们需要一个奇数尺寸的核，所以计算半径像素数，乘以2再加1
    kernel_radius_px = math.ceil(robot_radius / grid_size)
    kernel_size = 2 * kernel_radius_px + 1
    
    # -- 3. 准备输入张量 --
    # max_pool2d 需要一个 4D 的浮点型张量 [N, C, H, W]
    # 我们将输入的 [H, W] 转换为 [1, 1, H, W]
    grid_float_4d = occ_grid.float().unsqueeze(0).unsqueeze(0)
    
    # -- 4. 执行最大池化（等效于膨胀操作） --
    # padding='same' 或 kernel_size // 2 确保输出地图和输入地图尺寸一致
    # stride=1 意味着核会在每个像素上都滑动一次
    extended_grid_float = F.max_pool2d(
        grid_float_4d,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )

    # -- 5. 转换回布尔类型并返回 --
    # 池化后的结果中，任何非0的值都意味着原始区域内有障碍物
    extended_grid_bool = (extended_grid_float > 0.5).bool()
    
    # 将形状从 [1, 1, H, W] 转换回 [H, W]
    return extended_grid_bool.squeeze(0).squeeze(0)

def save_height_map(hm, output_path: str, cmap: str = "viridis", min_edge_px: int = 1024, dpi: int = 200) -> None:
    """
    将 height map 保存到本地，自动按网格尺寸放大到合适像素尺寸；若缺依赖则保存为 NPY。
    """
    if hm is None:
        print("WARNING: height_map is None; skip saving")
        return
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    arr = hm.detach().cpu().numpy()
    H, W = arr.shape[-2], arr.shape[-1]
    # 优先使用 matplotlib 保存并控制图像尺寸/插值
    try:
        import matplotlib.pyplot as plt
        # 目标较长边像素
        target_edge_px = max(min_edge_px, max(H, W))
        scale = target_edge_px / float(max(H, W))
        fig_w = (W * scale) / dpi
        fig_h = (H * scale) / dpi
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(arr, cmap=cmap, origin='lower', interpolation='nearest')
        ax.set_axis_off()
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"INFO: Saved height_map PNG to {output_path} (size ~ {int(W*scale)}x{int(H*scale)} px)")
        return
    except Exception as e:
        print(f"WARNING: matplotlib save failed ({e}); fallback to PIL/NPY")
    # 尝试 PIL 保存灰度
    try:
        from PIL import Image
        import numpy as np
        vmin = float(arr.min()) if arr.size > 0 else 0.0
        vmax = float(arr.max()) if arr.size > 0 else 1.0
        if vmax <= vmin:
            norm = (arr - vmin)
        else:
            norm = (arr - vmin) / (vmax - vmin)
        img = (np.clip(norm, 0.0, 1.0) * 255.0).astype("uint8")
        Image.fromarray(img).save(output_path)
        print(f"INFO: Saved height_map PNG (PIL) to {output_path}")
        return
    except Exception as e:
        print(f"WARNING: PIL save failed ({e}); fallback to NPY")
    # 最后退化为 NPY
    try:
        import numpy as np
        np.save(output_path if output_path.endswith('.npy') else output_path + '.npy', arr)
        print(f"INFO: Saved height_map NPY to {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save height_map to {output_path}: {e}")


def visualize_paths_on_grid(
    grid_map: torch.Tensor,
    waypoints: torch.Tensor,
    waypoint_lengths: torch.Tensor,
    bounds: tuple[float, float, float, float],
    grid_size: float,
    output_path: str = "planned_paths.png",
    max_trajs: int = 10,
    dpi: int = 200,
    min_edge_px: int = 1200,
) -> None:
    """
    Visualize first N trajectories over the occupancy grid and save to file.

    Args:
        grid_map: [H, W] bool or 0/1 tensor (True/1 means occupied)
        waypoints: [num_envs, max_wps, 3] world coordinates (x, y, z)
        waypoint_lengths: [num_envs] number of valid points per env
        bounds: (xmin, xmax, ymin, ymax)
        grid_size: meters per grid cell
        output_path: file path to save the image
        max_trajs: number of trajectories to draw
        dpi: figure dpi
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"WARNING: matplotlib not available for path visualization: {e}")
        return

    # convert grid map to numpy
    if isinstance(grid_map, torch.Tensor):
        gm = grid_map.detach().to("cpu").numpy()
    else:
        gm = grid_map
    gm = (gm > 0).astype(float)

    xmin, xmax, ymin, ymax = map(float, bounds)
    H, W = gm.shape

    # compute figure size based on target pixel edge
    target_edge_px = max(min_edge_px, max(H, W))
    scale = target_edge_px / float(max(H, W))
    fig_w = (W * scale) / dpi
    fig_h = (H * scale) / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    # show obstacles as black (1.0) and free as white (0.0)
    ax.imshow(gm, cmap="gray_r", origin="lower", interpolation="nearest")

    # colors for up to max_trajs
    from itertools import cycle
    color_cycle = cycle([
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    ])

    num_envs = int(waypoints.shape[0])
    draw_n = min(max_trajs, num_envs)

    # draw each trajectory
    for i in range(draw_n):
        n = int(waypoint_lengths[i].item())
        if n <= 0:
            continue
        pts = waypoints[i, :n, :2].detach().to("cpu").numpy()
        # map world -> grid indices
        gx = np.clip(((pts[:, 0] - xmin) / grid_size).astype(np.float32), 0, W - 1)
        gy = np.clip(((pts[:, 1] - ymin) / grid_size).astype(np.float32), 0, H - 1)
        c = next(color_cycle)
        # thinner lines/markers when figure is upscaled
        lw = max(0.6, 1.2 / max(scale, 1.0))
        ms_start = max(8.0, 24.0 / max(scale, 1.0))
        ms_goal = max(10.0, 28.0 / max(scale, 1.0))
        ax.plot(gx, gy, color=c, linewidth=lw, alpha=0.9)
        ax.scatter(gx[:1], gy[:1], color=c, marker="o", s=ms_start)
        ax.scatter(gx[-1:], gy[-1:], color=c, marker="x", s=ms_goal)

    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"INFO: Saved planned paths visualization to {output_path}")