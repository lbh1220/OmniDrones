import torch
import torch.nn.functional as F
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List

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
    show_grid: bool = True,
    grid_interval: float | None = None,
    grid_alpha: float = 0.25,
    grid_linewidth: float = 0.6,
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
    # show obstacles as black (1.0) and free as white (0.0) in WORLD coordinates
    # use extent to map array indices -> world (x,y)
    extent = [xmin, xmax, ymin, ymax]
    ax.imshow(gm, cmap="gray_r", origin="lower", interpolation="nearest", extent=extent, aspect='equal')

    # optional: draw coarse world-grid similar to convex hull visuals
    if show_grid:
        # default grid every 5 cells if not provided
        gi = float(grid_interval) if grid_interval is not None else 5.0 * float(grid_size)
        # major ticks in world coordinates
        xticks = np.arange(np.floor(xmin / gi) * gi, xmax + gi * 0.5, gi)
        yticks = np.arange(np.floor(ymin / gi) * gi, ymax + gi * 0.5, gi)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.grid(which="major", color="k", linestyle="-", alpha=float(grid_alpha), linewidth=float(grid_linewidth))
        # keep axis labels/ticks visible
        ax.tick_params(which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

    # colors for up to max_trajs
    from itertools import cycle
    color_cycle = cycle([
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    ])

    num_envs = int(waypoints.shape[0])
    draw_n = min(max_trajs, num_envs)

    # draw each trajectory (in world coordinates)
    for i in range(draw_n):
        n = int(waypoint_lengths[i].item())
        if n <= 0:
            continue
        pts = waypoints[i, :n, :2].detach().to("cpu").numpy()
        c = next(color_cycle)
        lw = 1.0
        ms_start = 18.0
        ms_goal = 22.0
        ax.plot(pts[:, 0], pts[:, 1], color=c, linewidth=lw, alpha=0.9)
        ax.scatter(pts[:1, 0], pts[:1, 1], color=c, marker="o", s=ms_start)
        ax.scatter(pts[-1:, 0], pts[-1:, 1], color=c, marker="x", s=ms_goal)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    # plt.close(fig)
    print(f"INFO: Saved planned paths visualization to {output_path}")


def get_convex_hulls_from_grid(
    occ_grid: torch.Tensor,
    grid_bounds: tuple[float, float, float, float],
    grid_size: float,
    min_area_m2: float = 0.5,
    simplification_tolerance_m: float = 0.1
) -> list[np.ndarray]:
    """
    从2D占据栅格地图中提取所有障碍物簇的简化凸包。
    
    使用OpenCV高效地执行：
    1. 查找轮廓 (findContours) 进行聚类。
    2. 计算凸包 (convexHull) 作为包络。
    3. 【新增】使用 (approxPolyDP) 简化凸包，去除锯齿状的小边。
    4. 将像素坐标转换回世界坐标。

    Args:
        occ_grid: 布尔型占据栅格，形状为 [H, W]。
        grid_bounds: 地图的物理边界 (xmin, xmax, ymin, ymax)。
        grid_size: 栅格地图的分辨率 (米/像素)。
        min_area_m2: 过滤掉的最小障碍物簇面积（平方米）。
        simplification_tolerance_m: 简化容差（米）。
            这是近似多边形与原始凸包之间的最大允许偏差。
            值越大，简化程度越高，凸包的边数越少。
            设为 0.0 可以跳过简化。

    Returns:
        一个列表，其中每个元素是一个 [N, 2] 的 NumPy 数组，
        代表一个【简化的】凸包的 N 个顶点在世界坐标系下的 (x, y) 坐标。
    """
    import cv2
    # -- 1. 准备输入：将 PyTorch 张量转换为 OpenCV 格式 --
    if occ_grid.dim() != 2:
        raise ValueError(f"Input occ_grid must be a 2D tensor [H, W], got {occ_grid.shape}")
    grid_np = (occ_grid.cpu().numpy() * 255).astype(np.uint8)

    # -- 2. 聚类：查找轮廓 --
    contours, _ = cv2.findContours(grid_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_hulls_world = []
    pc_xmin, _, pc_ymin, _ = grid_bounds
    min_area_px = min_area_m2 / (grid_size * grid_size)
    
    # 将简化容差从“米”转换为“像素”
    epsilon_px = simplification_tolerance_m / grid_size

    # -- 3. 提取、简化凸包并转换坐标 --
    for contour in contours:
        # a. 过滤掉太小的物体
        area = cv2.contourArea(contour)
        if area < min_area_px:
            continue
            
        # b. 计算精细的凸包
        hull_pixels_detailed = cv2.convexHull(contour)
        
        # c. 【新步骤】简化凸包
        if epsilon_px > 0.0 and len(hull_pixels_detailed) > 3:
            hull_pixels_simple = cv2.approxPolyDP(
                hull_pixels_detailed,
                epsilon=epsilon_px,  # 关键参数：像素单位的容差
                closed=True          # 这是一个闭合多边形
            )
        else:
            hull_pixels_simple = hull_pixels_detailed # 跳过简化

        # d. 检查简化后是否仍能构成多边形
        if len(hull_pixels_simple) < 3:
            continue
            
        # e. 转换坐标
        hull_pixels = hull_pixels_simple.squeeze(1)

        # -- 4. 将像素坐标 (x_pix, y_pix) 转换为世界坐标 (x_world, y_world) --
        hull_world_coords_x = pc_xmin + (hull_pixels[:, 0].astype(np.float32) + 0.5) * grid_size
        hull_world_coords_y = pc_ymin + (hull_pixels[:, 1].astype(np.float32) + 0.5) * grid_size
        hull_world_coords = np.stack([hull_world_coords_x, hull_world_coords_y], axis=1)

        all_hulls_world.append(hull_world_coords)

    return all_hulls_world

def plot_convex_hulls(
    hulls_list: List[np.ndarray],
    output_path: str,
    fill_color: str = 'gray',
    edge_color: str = 'black',
    alpha: float = 0.7,
    title: str = "Obstacle Convex Hulls"
):
    """
    将 get_convex_hulls_from_grid 生成的凸包列表绘制并保存为图像。
    
    Args:
        hulls_list: 一个列表，其中每个元素是 [N, 2] 的 NumPy 数组（世界坐标）。
        output_path: 保存图像的文件路径 (例如 "hulls.png")。
        fill_color: 凸包的填充颜色。
        edge_color: 凸包的边缘颜色。
        alpha: 填充的透明度。
        title: 图像的标题。
    """
    
    # 1. 创建一个绘图对象
    fig, ax = plt.subplots(figsize=(12, 12))

    if not hulls_list:
        print("Warning: No hulls provided to plot.")
    else:
        # 2. 遍历每一个凸包（每一个障碍物）
        for hull in hulls_list:
            if len(hull) < 2:
                continue # 跳过无效的点或线
                
            # 3. 【核心】手动闭合凸包
            #    将第一个顶点 [x0, y0] 附加到数组末尾
            #    hull (N, 2) -> closed_hull (N+1, 2)
            closed_hull = np.vstack([hull, hull[0]])
            
            # 4. 提取 X 和 Y 坐标
            x_coords = closed_hull[:, 0]
            y_coords = closed_hull[:, 1]
            
            # 5. 绘制填充的多边形
            ax.fill(
                x_coords, 
                y_coords, 
                facecolor=fill_color, 
                edgecolor=edge_color, 
                alpha=alpha, 
                linewidth=1.0
            )

    # 6. 设置绘图属性
    ax.set_xlabel("World X Coordinate (m)")
    ax.set_ylabel("World Y Coordinate (m)")
    ax.set_title(title)
    
    # 7. 【关键】设置相等的坐标轴比例
    #    这可以确保一个正方形不会被拉伸成一个矩形
    ax.set_aspect('equal', 'box')
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 8. 自动调整视图范围（如果提供了凸包）
    if hulls_list:
        all_points = np.vstack(hulls_list)
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        padding_x = (max_x - min_x) * 0.1
        padding_y = (max_y - min_y) * 0.1
        
        ax.set_xlim(min_x - padding_x - 1, max_x + padding_x + 1)
        ax.set_ylim(min_y - padding_y - 1, max_y + padding_y + 1)
        
    # 9. 保存并关闭图形
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Hull visualization saved to {output_path}")

