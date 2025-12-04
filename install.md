conda create -n drones41 python==3.10.12

conda activate drones41
pip install --upgrade pip # this one is important for following installations of isaac lab
# make sure the conda environment is activated by checking $CONDA_PREFIX
# then, at OmniDrones/
cp -r conda_setup/etc $CONDA_PREFIX
# re-activate the environment
conda activate drones41

# verification
python -c "from isaacsim import SimulationApp" # use omni.isaac.kit instead of isaacsim for isaac-sim-2022.*, isaac-sim-2023.*
# which torch is being used
python -c "import torch; print(torch.__path__)"


# The next step is to install Isaac Lab 

mkdir -p ~/my_lib
cd ~/my_lib

sudo apt install cmake build-essential

git clone https://github.com/isaac-sim/IsaacLab.git -b v1.1.0 IsaacLab_1

cd IsaacLab_1

# usd-core==23.11 is for nvidia-srl-usd 0.14.0, nvidia-srl-usd-to-urdf 0.6.0 requires usd-core <24.00, >=21.11
# lxml==4.9.4 is for nvidia-srl-usd-to-urdf 0.6.0 requires lxml <5.0.0, >=4.9.2
# tqdm is for nvidia-srl-usd 0.14.0 requires tqdm <5.0.0, >=4.63.0
# xxhash is for 50x faster cache checks
conda activate drones41
pip install usd-core==23.11 lxml==4.9.4 tqdm xxhash
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
# pip install onnx==1.16.2 # compatible with rsl_rl
pip install stable-baselines3==2.1.0 skrl==1.4.3
# install rsl_rl v2.0.1, cause isaac lab 1.1.0 requires rsl_rl v2.0.1
cd ~/my_lib
git clone https://github.com/leggedrobotics/rsl_rl.git -b v2.0.1
cd rsl_rl
pip install .

# in _isaac_lab/source/extensions/omni.isaac.lab_tasks/setup.py
# line 44 change to  "rsl-rl": []
# Install Isaac Lab
# at IsaacLab/
cd ~/my_lib/IsaacLab_1
./isaaclab.sh --install

cd ~/Projects/OmniTraffic

pip install -e .

# follow troubleshotting in omnidrones
# open /${HOME}/.local/share/ov/pkg/isaac-sim-4.1.0/exts/omni.isaac.core/omni/isaac/core/prims/xform_prim_view.py
# go to line 189 (for 4.1.0), 184 (for 4.0.0)

# L184
# default_positions, default_orientations = self.get_world_poses(usd=usd)

# remove usd=usd as shown below
# default_positions, default_orientations = self.get_world_poses()


# rvo
cd ~/my_lib
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
pip install Cython # 3.1.3
python setup.py build
python setup.py install