echo "Setup Isaac Sim Conda environment."
echo "Isaac Sim path: ${ISAACSIM_PATH}"

export PYTHONPATH_PREV=$PYTHONPATH
export LD_LIBRARY_PATH_PREV=$LD_LIBRARY_PATH

source ${ISAACSIM_PATH}/setup_conda_env.sh

# if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_CONNECTION" ]; then
#     echo "Connected via SSH."
#     if [ -z "$DISPLAY" ]; then
#         echo "Set DISPLAY=:10.0 to use X11 forwarding."
# 	export DISPLAY=:10.0
#     fi
# fi

# ---!! [START] DRLVO CUDNN 8.9.7 修复 !! ---
#
# 问题: 系统级的 cuDNN 8.9.7 (位于 /usr/local/cuda-11.8/lib) 
#      与 PyTorch 2.2.2 冲突, 导致 nn.Conv2d 崩溃。
# 方案: 强制将 Isaac Sim 预捆绑的、稳定的 cuDNN 8.7.0 
#      (位于 pip_prebundle 目录) 添加到 LD_LIBRARY_PATH 的最前端。
#
echo "DRLVO Fix: Prepending stable cuDNN 8.7.0 to LD_LIBRARY_PATH..."

# 2. 定义 "好" cuDNN 库的路径
#    (我们直接使用 ISAACSIM_PATH 变量来定位它)
export CORRECT_CUDNN_PATH="${ISAACSIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cudnn/lib"

# 3. 将 "好" 路径置顶 (最高优先级)
#    这会覆盖掉系统级的 "坏" 8.9.7 库。
export LD_LIBRARY_PATH=${CORRECT_CUDNN_PATH}:${LD_LIBRARY_PATH}