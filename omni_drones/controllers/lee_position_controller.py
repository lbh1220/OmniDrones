# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch._tensor import Tensor
import torch.nn as nn
from tensordict import TensorDict
from .controller import ControllerBase

from omni_drones.utils.torch import (
    quat_mul,
    quat_rotate_inverse,
    normalize,
    quaternion_to_rotation_matrix,
    quaternion_to_euler,
    axis_angle_to_quaternion,
    axis_angle_to_matrix
)
import yaml
import os.path as osp


def compute_parameters(
    rotor_config,
    inertia_matrix,
):
    rotor_angles = torch.as_tensor(rotor_config["rotor_angles"])
    arm_lengths = torch.as_tensor(rotor_config["arm_lengths"])
    force_constants = torch.as_tensor(rotor_config["force_constants"])
    moment_constants = torch.as_tensor(rotor_config["moment_constants"])
    directions = torch.as_tensor(rotor_config["directions"])
    max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])
    A = torch.stack(
        [
            torch.sin(rotor_angles) * arm_lengths,
            -torch.cos(rotor_angles) * arm_lengths,
            -directions * moment_constants / force_constants,
            torch.ones_like(rotor_angles),
        ]
    )
    mixer = A.T @ (A @ A.T).inverse() @ inertia_matrix

    return mixer

class LeePositionController(ControllerBase):
    """
    Computes rotor commands for the given control target using the controller
    described in https://arxiv.org/abs/1003.2005.

    Inputs:
        * root_state: tensor of shape (13,) containing position, rotation (in quaternion),
        linear velocity, and angular velocity.
        * control_target: tensor of shape (7,) contining target position, linear velocity,
        and yaw angle.

    Outputs:
        * cmd: tensor of shape (num_rotors,) containing the computed rotor commands.
        * controller_state: empty dict.
    """
    def __init__(
        self,
        g: float,
        uav_params,
    ) -> None:
        super().__init__()
        controller_param_path = osp.join(
            osp.dirname(__file__), "cfg", f"lee_controller_{uav_params['name']}.yaml"
        )
        with open(controller_param_path, "r") as f:
            controller_params = yaml.safe_load(f)

        self.pos_gain = nn.Parameter(torch.as_tensor(controller_params["position_gain"]).float())
        self.vel_gain = nn.Parameter(torch.as_tensor(controller_params["velocity_gain"]).float())
        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.g = nn.Parameter(torch.tensor([0.0, 0.0, g]).abs())

        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]

        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)

        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )
        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.attitute_gain = nn.Parameter(
            torch.as_tensor(controller_params["attitude_gain"]).float() @ I[:3, :3].inverse()
        )
        self.ang_rate_gain = nn.Parameter(
            torch.as_tensor(controller_params["angular_rate_gain"]).float() @ I[:3, :3].inverse()
        )
        self.requires_grad_(False)

    def compute(
        self,
        root_state: torch.Tensor,
        target_pos: torch.Tensor=None,
        target_vel: torch.Tensor=None,
        target_acc: torch.Tensor=None,
        target_yaw: torch.Tensor=None,
        body_rate: bool=False
    ):
        batch_shape = root_state.shape[:-1]
        device = root_state.device
        if target_pos is None:
            target_pos = root_state[..., :3]
        else:
            target_pos = target_pos.expand(batch_shape+(3,))
        if target_vel is None:
            target_vel = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_vel = target_vel.expand(batch_shape+(3,))
        if target_acc is None:
            target_acc = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_acc = target_acc.expand(batch_shape+(3,))
        if target_yaw is None:
            target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1]
        else:
            if not target_yaw.shape[-1] == 1:
                target_yaw = target_yaw.unsqueeze(-1)
            target_yaw = target_yaw.expand(batch_shape+(1,))

        cmd = self._compute(
            root_state.reshape(-1, 13),
            target_pos.reshape(-1, 3),
            target_vel.reshape(-1, 3),
            target_acc.reshape(-1, 3),
            target_yaw.reshape(-1, 1),
            body_rate
        )

        return cmd.reshape(*batch_shape, -1)

    def _compute(self, root_state, target_pos, target_vel, target_acc, target_yaw, body_rate):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_rate:
            # convert angular velocity from world frame to body frame
            ang_vel = quat_rotate_inverse(rot, ang_vel)

        pos_error = pos - target_pos
        vel_error = vel - target_vel

        acc = (
            pos_error * self.pos_gain
            + vel_error * self.vel_gain
            - self.g
            - target_acc
        )
        R = quaternion_to_rotation_matrix(rot)
        b1_des = torch.cat([
            torch.cos(target_yaw),
            torch.sin(target_yaw),
            torch.zeros_like(target_yaw)
        ],dim=-1)
        b3_des = -normalize(acc)
        b2_des = normalize(torch.cross(b3_des, b1_des, 1))
        R_des = torch.stack([
            b2_des.cross(b3_des, 1),
            b2_des,
            b3_des
        ], dim=-1)
        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R)
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        ang_error = torch.stack([
            ang_error_matrix[:, 2, 1],
            ang_error_matrix[:, 0, 2],
            ang_error_matrix[:, 1, 0]
        ],dim=-1)
        ang_rate_err = ang_vel
        ang_acc = (
            - ang_error * self.attitute_gain
            - ang_rate_err * self.ang_rate_gain
            + torch.linalg.cross(ang_vel, ang_vel)
        )
        thrust = (-self.mass * (acc * R[:, :, 2]).sum(-1, True))
        ang_acc_thrust = torch.cat([ang_acc, thrust], dim=-1)
        cmd = (self.mixer @ ang_acc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        return cmd

    def process_rl_actions(self, actions) -> Tensor:
        target_vel, target_yaw = actions.split([3, 1], dim=-1)
        return target_vel, target_yaw * torch.pi

class PlanarSpeedController(LeePositionController):
    """
    一个用于控制无人机在2D平面上运动的专用控制器。

    该控制器继承自 LeePositionController，旨在维持一个固定的目标高度，
    同时根据用户提供的2D速度指令（在XY平面上）进行移动。
    它会自动计算维持高度所需的Z轴速度。
    """
    def __init__(
        self,
        g: float,
        uav_params,
        # 新增一个用于高度控制的增益参数
        height_p_gain: float = 2.0,
        height_d_gain: float = 1.0   # D增益 (需要调优)
    ) -> None:
        # 首先，调用父类的构造函数来初始化所有底层参数（如 pos_gain, mass, mixer 等）
        super().__init__(g, uav_params)
        
        # 保存高度控制的 P 增益
        self.height_p_gain = nn.Parameter(torch.tensor(height_p_gain))
        self.height_d_gain = nn.Parameter(torch.tensor(height_d_gain))
        self.requires_grad_(False)

    def compute(
        self,
        root_state: torch.Tensor,
        target_vel_xy: torch.Tensor,
        target_height: torch.Tensor,
        target_yaw: torch.Tensor = None
    ):
        """
        计算旋翼指令以维持目标高度并遵循2D目标速度。

        Args:
            root_state (torch.Tensor): 机器人的完整状态 [batch_size, 13]。
            target_vel_xy (torch.Tensor): 在XY平面上的目标速度 [batch_size, 2]。
            target_height (torch.Tensor): 目标高度Z值 [batch_size, 1]。
            target_yaw (torch.Tensor, optional): 目标偏航角 [batch_size, 1]。默认为 None，表示维持当前角度。

        Returns:
            torch.Tensor: 计算出的旋翼指令 [batch_size, num_rotors]。
        """
        # 1. 从完整状态中获取当前位置
        current_pos = root_state[..., :3]
        current_height = current_pos[..., 2:3]

        current_vel_z = root_state[..., 9:10] # 获取当前Z轴速度 (在世界坐标系)

        # 2. 【关键步骤 2】我们自己计算Z轴PD控制
        height_error = target_height - current_height
        
        # P 项 (比例)：修正当前的位置误差
        p_term = height_error * self.height_p_gain
        
        # D 项 (微分)：抵抗当前的Z轴速度 (增加阻尼)
        d_term = -self.height_d_gain * current_vel_z 
        
        vel_z_correction = (p_term + d_term)
        
        # 提高修正速度的上限 (例如 +/- 5m/s)，这需要调优
        vel_z_correction = vel_z_correction.clamp(-5.0, 5.0)
        # vel_z_correction = torch.zeros_like(vel_z_correction)

        # 3. 构造一个完整的3D目标速度
        #    将用户输入的XY速度和我们计算出的Z速度合并
        target_vel_3d = torch.cat([target_vel_xy, vel_z_correction], dim=-1)

        # 4. (可选但推荐) 构造一个辅助的目标位置
        #    我们告诉底层控制器，XY方向的目标就是当前位置（因为我们主要控制速度），
        #    而Z方向的目标是我们的目标高度。这能辅助底层控制器更好地维持高度。
        target_pos_3d = current_pos.clone()
        # target_pos_3d[..., 2:3] = target_height

        # 5. 调用父类的 compute 方法
        #    我们已经准备好了所有“翻译”过的、底层控制器能理解的3D指令。
        #    现在，让 LeePositionController 的原始实现去处理剩下的所有复杂物理计算。
        return super().compute(
            root_state=root_state,
            target_pos=target_pos_3d,
            target_vel=target_vel_3d,
            target_acc=None,  # 我们不控制加速度，让父类处理
            target_yaw=target_yaw,
            body_rate=False
        )


class AttitudeController(ControllerBase):
    r"""

    """
    def __init__(self, g, uav_params):
        super().__init__()
        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.g = nn.Parameter(torch.tensor(g))
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )

        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.gain_attitude = nn.Parameter(
            torch.tensor([3., 3., 0.035]) @ I[:3, :3].inverse()
        )
        self.gain_angular_rate = nn.Parameter(
            torch.tensor([0.52, 0.52, 0.025]) @ I[:3, :3].inverse()
        )


    def forward(
        self,
        root_state: torch.Tensor,
        target_thrust: torch.Tensor,
        target_yaw_rate: torch.Tensor=None,
        target_roll: torch.Tensor=None,
        target_pitch: torch.Tensor=None,
    ):
        batch_shape = root_state.shape[:-1]
        device = root_state.device

        if target_yaw_rate is None:
            target_yaw_rate = torch.zeros(*batch_shape, 1, device=device)
        if target_pitch is None:
            target_pitch = torch.zeros(*batch_shape, 1, device=device)
        if target_roll is None:
            target_roll = torch.zeros(*batch_shape, 1, device=device)

        cmd = self._compute(
            root_state.reshape(-1, 13),
            target_thrust.reshape(-1, 1),
            target_yaw_rate=target_yaw_rate.reshape(-1, 1),
            target_roll=target_roll.reshape(-1, 1),
            target_pitch=target_pitch.reshape(-1, 1),
        )
        return cmd.reshape(*batch_shape, -1)

    def _compute(
        self,
        root_state: torch.Tensor,
        target_thrust: torch.Tensor,
        target_yaw_rate: torch.Tensor,
        target_roll: torch.Tensor,
        target_pitch: torch.Tensor
    ):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        device = pos.device

        R = quaternion_to_rotation_matrix(rot)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).unsqueeze(-1)
        yaw = axis_angle_to_matrix(yaw, torch.tensor([0., 0., 1.], device=device))
        roll = axis_angle_to_matrix(target_roll, torch.tensor([1., 0., 0.], device=device))
        pitch = axis_angle_to_matrix(target_pitch, torch.tensor([0., 1., 0.], device=device))
        R_des = torch.bmm(torch.bmm(yaw,  roll), pitch)
        angle_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R)
            - torch.bmm(R.transpose(-2, -1), R_des)
        )

        angle_error = torch.stack([
            angle_error_matrix[:, 2, 1],
            angle_error_matrix[:, 0, 2],
            torch.zeros(yaw.shape[0], device=device)
        ], dim=-1)

        angular_rate_des = torch.zeros_like(ang_vel)
        angular_rate_des[:, 2] = target_yaw_rate.squeeze(1)
        angular_rate_error = ang_vel - torch.bmm(torch.bmm(R_des.transpose(-2, -1), R), angular_rate_des.unsqueeze(2)).squeeze(2)

        angular_acc = (
            - angle_error * self.gain_attitude
            - angular_rate_error * self.gain_angular_rate
            + torch.linalg.cross(ang_vel, ang_vel)
        )
        angular_acc_thrust = torch.cat([angular_acc, target_thrust], dim=1)
        cmd = (self.mixer @ angular_acc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        return cmd


class RateController(ControllerBase):

    def __init__(self, g, uav_params) -> None:
        super().__init__()
        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.g = nn.Parameter(torch.tensor(g))
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )

        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.gain_angular_rate = nn.Parameter(
            torch.tensor([0.52, 0.52, 0.025]) @ I[:3, :3].inverse()
        )


    def forward(
        self,
        root_state: torch.Tensor,
        target_rate: torch.Tensor,
        target_thrust: torch.Tensor,
    ):
        assert root_state.shape[:-1] == target_rate.shape[:-1]

        batch_shape = root_state.shape[:-1]
        root_state = root_state.reshape(-1, 13)
        target_rate = target_rate.reshape(-1, 3)
        target_thrust = target_thrust.reshape(-1, 1)

        pos, rot, linvel, angvel = root_state.split([3, 4, 3, 3], dim=1)
        body_rate = quat_rotate_inverse(rot, angvel)

        rate_error = body_rate - target_rate
        acc_des = (
            - rate_error * self.gain_angular_rate
            + angvel.cross(angvel)
        )
        angacc_thrust = torch.cat([acc_des, target_thrust], dim=1)
        cmd = (self.mixer @ angacc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        cmd = cmd.reshape(*batch_shape, -1)
        return cmd

    def process_rl_actions(self, actions: torch.Tensor):
        target_rate, target_thrust = actions.split([3, 1], -1)
        target_thrust = ((target_thrust + 1) / 2).clip(0.) * self.max_thrusts
        return target_rate * torch.pi, target_thrust

