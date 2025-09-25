# a) 导入必要的模块
import omni.isaac.lab.sim as sim_utils
from dataclasses import dataclass, field
# This part could not be used now in current isaac lab env
# b) 定义 PathCfg 类
@dataclass
class PathCfg:
    """
    一个独立的、用来描述路径线条的配置类。
    我们直接定义所有需要的属性，而不是通过继承获得。
    """
    # prim_path 是必需的，所以我们不给它默认值
    prim_path: str
    
    # 线条的宽度
    width: float = 0.05
    
    # 视觉材质，用于定义颜色等。
    # 我们使用 field(default_factory=...) 来提供一个默认的材质实例。
    visual_material: sim_utils.VisualMaterialCfg = field(
        default_factory=lambda: sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
    )


# c) 导入USD和isaac sim相关的API
from pxr import UsdGeom, Gf, Vt
import torch
import numpy as np

class PathVisualizer:
    """一个使用 UsdGeom.BasisCurves 来高效可视化路径的工具类。"""

    def __init__(self, cfg: PathCfg):
        """
        初始化并根据配置在场景中创建一个 BasisCurves prim。
        
        Args:
            cfg: 我们定义的 PathCfg 实例。
        """
        self.cfg = cfg
        self._stage = sim_utils.get_current_stage()
        
        # 1. 在指定的 prim_path 创建或获取 BasisCurves prim
        self._curves = UsdGeom.BasisCurves.Define(self._stage, self.cfg.prim_path)
        
        # 2. 设置曲线的静态属性（这些通常只需要设置一次）
        #    设置为 "linear"，意味着点与点之间是直线连接
        self._curves.GetTypeAttr().Set("linear")
        
        #    设置曲线的宽度
        widths_attr = self._curves.GetWidthsAttr()
        #    BasisCurves 的宽度是每个点一个值，但我们可以用 "vertex" 插值来让整条线宽度一致
        widths_attr.Set(Vt.FloatArray([self.cfg.width]))
        self._curves.GetWidthsInterpolationAttr().Set(UsdGeom.Tokens.vertex)
        
        # 3. 应用材质来设置颜色（复用 Isaac Lab 的功能）
        if self.cfg.visual_material:
            sim_utils.apply_visual_material(self._curves.GetPrim(), self.cfg.visual_material)
            
    def update(self, path_points: torch.Tensor | np.ndarray):
        """
        用一组新的点来更新路径。
        
        Args:
            path_points: 一个形状为 (N, 3) 的张量或数组，N是路径上的点的数量。
        """
        if not self._curves.GetPrim().IsValid():
            print(f"警告: BasisCurves prim at {self.cfg.prim_path} 已失效。")
            return
            
        num_points = path_points.shape[0]
        if num_points == 0:
            # 如果没有点，设置一个空数组来清空路径
            self._curves.GetPointsAttr().Set(Vt.Vec3fArray())
            self._curves.GetCurveVertexCountsAttr().Set(Vt.IntArray())
            return

        # 1. 转换数据格式为 USD 所需的 Vt.Vec3fArray
        if isinstance(path_points, torch.Tensor):
            points_data = path_points.detach().cpu().numpy()
        else:
            points_data = path_points
        
        # 2. 设置曲线的顶点位置
        self._curves.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points_data))
        
        # 3. 设置 "顶点计数"，这告诉USD我们只有一条曲线，它包含了所有的N个顶点
        self._curves.GetCurveVertexCountsAttr().Set(Vt.IntArray([num_points]))

    def set_visibility(self, visible: bool):
        """设置路径的可见性。"""
        imageable = UsdGeom.Imageable(self._curves.GetPrim())
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()