import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a grid with optional subtitles and title."""
    # show images in a grid
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    
    # [修复] 处理只有一个子图的情况，将其转为数组以便后续统一处理
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    # plot images
    for idx, (img, ax) in enumerate(zip(images, axes)):
        # 确保只处理存在的图片（如果 grid 空位多于图片数）
        if idx < n_images:
            img = img.detach().cpu().numpy()
            ax.imshow(img, cmap=cmap)
            ax.axis("off")
            if subtitles:
                ax.set_title(subtitles[idx])
        else:
            # 隐藏多余的坐标轴
            ax.axis("off")
            
    # remove extra axes if any (处理多出的空白格)
    for ax in axes[n_images:]:
        fig.delaxes(ax)
        
    # set title
    if title:
        plt.suptitle(title)

    # adjust layout to fit the title
    plt.tight_layout()
    # save the figure
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    # close the figure
    plt.close()