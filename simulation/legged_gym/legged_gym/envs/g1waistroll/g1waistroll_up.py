# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This file was modified by HumanUP authors in 2024-2025
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: # Copyright (c) 2021 ETH Zurich, Nikita Rudin. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024-2025 RoboVision Lab, UIUC. All rights reserved.

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from tqdm import tqdm
from warnings import WarningMessage
import numpy as np
import os
import cv2

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from legged_gym.gym_utils.terrain import Terrain
from legged_gym.gym_utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.gym_utils.helpers import class_to_dict
from legged_gym.envs.base.humanoid import Humanoid
from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO
from legged_gym.envs.g1waist.g1waist_up_config import G1WaistHumanUPCfg

def quat_distance_rad(q1, q2):
    dot = torch.sum(q1 * q2, dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2 * torch.acos(dot)  # rad

class G1WaistRollHumanUP(Humanoid):
    def __init__(self, cfg: G1WaistHumanUPCfg, sim_params, physics_engine, sim_device, headless):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.domain_rand_general = self.cfg.domain_rand.domain_rand_general

        # Pre init for motion loading
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        # xyz, quat, lin_vel, ang_vel
        self.initial_root_states = torch.tensor([
            8.6570e-03,  5.0515e-04,  5.6526e-02+0.02, 
            -9.8234e-03,  4.9986e-01,  1.7525e-02, -8.6587e-01,
            1.0610e-04, -4.5519e-05,  2.4261e-03,
            5.6199e-03, -8.3706e-03, -1.0773e-03]).to(sim_device).repeat(self.num_envs, 1)
        self.initial_root_states_face_down = torch.tensor([ 
            8.6570e-03,  5.0515e-04,  5.6526e-02+0.02, 
            -0.0175, -0.8659, -0.0098, -0.4999,
            1.0610e-04, -4.5519e-05,  2.4261e-03,
            5.6199e-03, -8.3706e-03, -1.0773e-03]).to(sim_device).repeat(self.num_envs, 1)
        self.initial_dof_pos = torch.tensor([-0.3600,  0.2481,  1.6115, -0.0647, -0.8612, -0.1226, -0.3878,  0.3584,
            1.5328,  0.1519, -0.8651,  0.2362, -0.0357,  0.0685, -0.5200,  0.4665,
            0.8218,  0.4253,  1.2972,  0.1429, -1.0324, -0.4241,  1.4075]).to(sim_device).repeat(self.num_envs, 1)
        
        self.left_dof_indices = torch.tensor([0, 1, 2, 3, 4, 5, 15, 16, 17, 18], device=self.device, dtype=torch.long)
        self.right_dof_indices = torch.tensor([6, 7, 8, 9, 10, 11, 19, 20, 21, 22], device=self.device, dtype=torch.long)
        self.waist_indices = torch.tensor([12, 13], device=self.device, dtype=torch.long)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()

        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.termination_height = torch.zeros(self.num_envs, device=self.device)  # NOTE: This is for curriculum recording
        self.termination_gravity = torch.zeros(self.num_envs, device=self.device)  # NOTE
        self.termination_quat_error = torch.zeros(self.num_envs, device=self.device)  # NOTE

        # fall down states
        self._recovery_episode_prob = 0.5
        self._recovery_steps = 150
        self._fall_init_prob = 0.5

        self.standing_init_prob = cfg.rewards.standing_scale_range[1]  # NOTE
        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)
        self.post_physics_step()


        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

    def _init_buffers(self):
        super()._init_buffers()
        self.rigid_body_rot = self.rigid_body_states[..., :self.num_bodies, 3:7]

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
            2.1 creates the environment,
            2.2 calls DOF and Rigid shape properties callbacks,
            2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
        self.torso_idx = self.gym.find_asset_rigid_body_index(
            robot_asset, self.cfg.asset.torso_name
        )
        self.chest_idx = self.gym.find_asset_rigid_body_index(
            robot_asset, self.cfg.asset.chest_name
        )
        self.head_idx = self.gym.find_asset_rigid_body_index(
            robot_asset, self.cfg.asset.forehead_name
        )

        for s in self.cfg.asset.feet_bodies:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []

        # record the initial standing default state
        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        spacing = self.cfg.env.env_spacing
        if self.cfg.terrain.mesh_type == "plane":
            env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
            env_upper = gymapi.Vec3(spacing, spacing, spacing)
        else:
            env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
            env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(
                    0.0, 0.0, self.cfg.env.rand_yaw_range * np.random.uniform(-1, 1)
                )
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, anymal_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

            self.mass_params_tensor[i, :] = (
                torch.from_numpy(mass_params).to(self.device).to(torch.float)
            )
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = (
                self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
            )

        self.body_names = body_names
        self._get_body_indices()

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )
        self.knee_indices = torch.zeros(
            len(knee_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], knee_names[i]
            )

        self.penalized_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        if self.cfg.env.record_video:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720 * 2
            camera_props.height = 480 * 2
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                cam_pos = np.array([2, 0, 0.3])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(
                    camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0 * cam_pos)
                )

    def _reset_dofs(self, env_ids, dof_pos=None, dof_vel=None, set_act=True):
        if dof_pos is None:
            self.dof_pos[env_ids] = self.initial_dof_pos[env_ids].clone()
        else:
            self.dof_pos[env_ids] = dof_pos[env_ids].clone()
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        if set_act:
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
        )
    def _reset_root_states(self, env_ids, root_vel=None, root_quat=None, root_height=None, use_base_init_state=False, set_act=True):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if use_base_init_state:
            self.root_states[env_ids] = self.base_init_state
        else:
            self.root_states[env_ids] = self.initial_root_states_face_down[env_ids].clone()
        if self.custom_origins:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 2] += 0.01
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(
                    -0.3, 0.3, (len(env_ids), 2), device=self.device
                )  # xy position within 1m of the center
        else:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        if set_act is True:
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

    def _reset_stand_and_lie_states(self, env_ids, dof_pos):
        if self.cfg.rewards.standing_scale_curriculum:
            self._update_standing_prob_curriculum() 
        num_standing = int(self.num_envs * self.standing_init_prob)
        # sample_standing_env_idx = torch.randperm(self.num_envs)[:num_standing]
        standing_env_flag = env_ids < num_standing
        non_standing_env_flag = env_ids >= num_standing

        # reset robot states
        # falling states
        if len(env_ids[non_standing_env_flag]) > 0:
            self._reset_dofs(env_ids[non_standing_env_flag], set_act=False)
            self._reset_root_states(env_ids[non_standing_env_flag], set_act=False)
        # standing states
        if len(env_ids[standing_env_flag]) > 0:
            self._reset_dofs(env_ids[standing_env_flag], dof_pos=dof_pos, set_act=False)
            self._reset_root_states(env_ids[standing_env_flag], use_base_init_state=True, set_act=False)

        env_ids_int32 = env_ids.to(dtype=torch.int32)  # NOTE: This is the env_ids for the entire envs
        # reset robot states
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        # reset robot dofs
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )


    def _update_standing_prob_curriculum(self):
        # [Curriculum] Update the standing probability based on the curriculum type
        assert self.cfg.rewards.standing_scale_curriculum_type in ["sin", "step_height", "step", "cos"]
        if self.cfg.rewards.standing_scale_curriculum_type == "sin":
            # Sine wave curriculum, the standing probability will change from 0 to 1 and back to 0 in a cycle
            iteration = self.total_env_steps_counter // 24
            cycle_iteration = iteration % self.cfg.rewards.standing_scale_curriculum_iterations
            sin_progress = (cycle_iteration / self.cfg.rewards.standing_scale_curriculum_iterations) * torch.pi
            self.cfg.rewards.standing_scale = self.cfg.rewards.standing_scale_range[0] + (
                self.cfg.rewards.standing_scale_range[1] - self.cfg.rewards.standing_scale_range[0]
            ) * torch.sin(torch.tensor(sin_progress).to(self.device))
            self.standing_init_prob = self.cfg.rewards.standing_scale.clamp(0.0, 1.0)
        elif self.cfg.rewards.standing_scale_curriculum_type == "cos":
            # Cosine annealing curriculum, the standing probability will decrease from 1 to 0
            iteration = self.total_env_steps_counter // 24
            cycle_iteration = iteration % self.cfg.rewards.standing_scale_curriculum_iterations
            cos_progress = (cycle_iteration / self.cfg.rewards.standing_scale_curriculum_iterations) * torch.pi / 2.0
            self.cfg.rewards.standing_scale = self.cfg.rewards.standing_scale_range[0] + (
                self.cfg.rewards.standing_scale_range[1] - self.cfg.rewards.standing_scale_range[0]
            ) * torch.cos(torch.tensor(cos_progress).to(self.device)).clamp(0.0, 1.0)
            self.standing_init_prob = self.cfg.rewards.standing_scale
        elif self.cfg.rewards.standing_scale_curriculum_type == "step_height":
            # Step height curriculum, the standing probability will change based on the average termination height of all envs
            # NOTE: This is an inversed version of regularization scale curriculum in step_height - Runpei
            if torch.mean(self.termination_height).item() > 0.65:
                # drease the regularization scale
                self.standing_init_prob *= (
                    1.0 - self.cfg.rewards.standing_scale_gamma
                )
            elif torch.mean(self.termination_height).item() < 0.1:
                # increase the regularization scale
                self.standing_init_prob *= (
                    1.0 + self.cfg.rewards.standing_scale_gamma
                )
            self.standing_init_prob = max(
                min(
                    self.standing_init_prob,
                    self.cfg.rewards.standing_scale_range[1],
                ),
                self.cfg.rewards.standing_scale_range[0],
            )
        elif self.cfg.rewards.standing_scale_curriculum_step:
            # TODO: implement step curriculum - Runpei
            raise NotImplementedError


    def reset_idx(self, env_ids, init=False):
        if len(env_ids) == 0:
            return

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        dof_pos = self.default_dof_pos_all.clone()

        if self.standing_init_prob > 0:
            self._reset_stand_and_lie_states(env_ids, dof_pos=dof_pos)
        else:
            # reset robot states
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._resample_commands(env_ids)  # no resample commands
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_torques[env_ids] = 0.0
        self.last_root_vel[:] = 0.0
        self.last_contact_forces[env_ids] = torch.zeros_like(self.contact_forces[env_ids])
        self.last_base_height = torch.zeros_like(self.root_states[env_ids, 2])
        self.feet_air_time[env_ids] = 0.0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.0
        self.contact_buf[env_ids, :, :] = 0.0
        self.action_history_buf[env_ids, :, :] = 0.0
        self.feet_land_time[env_ids] = 0.0
        self._reset_buffers_extra(env_ids)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["metric_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids] * self.reward_scales[key])
                / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return

    def _update_recovery_count(self):
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())

        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.cfg.domain_rand.push_interval == 0
        ):
            self._push_robots()

        if self.cfg.domain_rand.drag_robot_up:
            # TODO: Fix how to drag the torso instead of the base
            if self.cfg.domain_rand.drag_when_falling:
                # drag the robot when the base is going down
                if self.cfg.domain_rand.force_compenstation:
                    # self._drag_robots_by_force(self.cfg.domain_rand.drag_force)
                    # self._drag_robots_torso_body(self.base_lin_vel[:, 2], random=False)
                    self._drag_robots(self.base_lin_vel[:, 2], random=False)
                else:
                    # self._drag_robots_torso_body(self.base_lin_vel[:, 2])
                    if self.cfg.domain_rand.drag_robot_by_force:
                        self._drag_robots_by_force()
                    else:
                        self._drag_robots(self.base_lin_vel[:, 2])
            elif self.common_step_counter % self.cfg.domain_rand.drag_interval == 0:
                # drag the robot up regularly
                if self.cfg.domain_rand.drag_robot_by_force:
                    self._drag_robots_by_force()
                # self._drag_robots_torso_body()  # TODO: implement it
                else:
                    self._drag_robots()

    def _drag_robots(self, z_vel=None, random=True):
        """Random drags the robots up. Emulates an impulse by setting a randomized base upper velocity."""
        if z_vel is None:
            min_drag_vel = self.cfg.domain_rand.min_drag_vel
            max_drag_vel = self.cfg.domain_rand.max_drag_vel
            self.root_states[:, 9] += torch_rand_float(min_drag_vel, max_drag_vel, (self.num_envs,1), device=self.device).squeeze(1)  # lin vel z
        else:
            drag_flag = z_vel < 0
            if random:
                self.root_states[drag_flag, 9] = torch_rand_float(
                    min_drag_vel, max_drag_vel, (self.num_envs,), device=self.device
                )  # lin vel z
            else:
                self.root_states[drag_flag, 9] = -z_vel[drag_flag]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_drag_force_curriculum(self, force):
        base_height = self.root_states[:, 2]
        target_height = self.cfg.domain_rand.drag_force_curriculum_target_height
        assert self.cfg.domain_rand.drag_force_curriculum_type in ["linear", "sin"]
        if self.cfg.domain_rand.drag_force_curriculum_type == "linear":
            # linearly decrease the drag force based on the robot's height
            force = force * torch.clamp(1 - base_height / target_height, min=0.0, max=1.0)
        elif self.cfg.domain_rand.drag_force_curriculum_type == "sin":
        # elif self.cfg.domain_rand.drag_force_curriculum_cosine_height:
            # cosine decrease the drag force based on the robot's height (0, pi/2) ~ (0, 1)
            # NOTE: This would decrase the force to 0 at the target height faster than linear mode
            sin_progress = (base_height / target_height) * torch.pi / 2.0
            force = force * (1 - torch.sin(sin_progress))
        else:
            raise NotImplementedError
        return force

    def _drag_robots_by_force(self):
        """Drag the robots up by applying a force"""
        # TODO: use a force curriculum based on the robot's height
        if self.cfg.domain_rand.drag_force_curriculum:
            force = self._update_drag_force_curriculum(self.cfg.domain_rand.drag_force)
        else:
            force = self.cfg.domain_rand.drag_force
        forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        assert self.cfg.domain_rand.drag_robot_part in ["torso", "chest", "head"], "Now only support dragging torso, chest, or head"
        forces[:, eval("self." + self.cfg.domain_rand.drag_robot_part + "_idx"), 2] = force
        # forces[:, self.torso_idx, 2] = force
        # torques[:, self.torso_idx, 2] = 0  # torque (used to rotate the body), we don't need it -- Runpei
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

    def check_termination(self):
        super().check_termination()

        if self.cfg.env.terminate_on_velocity:
            base_vel = torch.norm(self.base_lin_vel, dim=-1)
            vel_too_large = base_vel > 2.5
            self.reset_buf[vel_too_large] = 1
 
        if self.cfg.env.terminate_on_height:
            base_too_high = torch.logical_or(self.root_states[:, 2] > 1.2, self.root_states[:, 2] < 0.0)
            self.reset_buf[base_too_high] = 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.termination_height[env_ids] = self.root_states[env_ids, 2]
            target_gravity = torch.tensor([-9.81, 0, 0], device=self.device, dtype=torch.float)
            self.termination_gravity[env_ids] = 1 - torch.nn.functional.cosine_similarity(self.projected_gravity[env_ids], target_gravity, dim=-1)
            target_quat = [0.7071, 0, 0.7071, 0]
            target_quat = torch.tensor(target_quat, device=self.device, dtype=torch.float)
            self.termination_quat_error = quat_distance_rad(self.base_quat, target_quat)

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        # Humanoid.post_physics_step(self)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.episode_length[env_ids] = self.episode_length_buf[env_ids].float()

        self.reset_idx(env_ids)

        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13].clone()
        self.last_base_height = self.root_states[:, 2].clone()
        self.last_contact_forces = self.contact_forces.clone()

        if self.cfg.rewards.regularization_scale_curriculum:
            self._update_regularization_scale_curriculum()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_goal()

        self._update_recovery_count()

    def _update_regularization_scale_curriculum(self):
        # [Curriculum] Regularization scale curriculum
        assert self.cfg.rewards.regularization_scale_curriculum_type in ["sin", "step_height", "step_episode"]
        if self.cfg.rewards.regularization_scale_curriculum_type == "sin":
            # Cycled cosine annealing
            iteration = self.total_env_steps_counter // 24
            cycle_iteration = iteration % self.cfg.rewards.regularization_scale_curriculum_iterations
            sin_progress = (cycle_iteration / self.cfg.rewards.regularization_scale_curriculum_iterations) * torch.pi
            self.cfg.rewards.regularization_scale = self.cfg.rewards.regularization_scale_range[0] + (
                self.cfg.rewards.regularization_scale_range[1] - self.cfg.rewards.regularization_scale_range[0]
            ) * torch.sin(torch.tensor(sin_progress).to(self.device)).clamp(0.0, 1.0)
        elif self.cfg.rewards.regularization_scale_curriculum_type == "step_height":
            # Step curriculum based on the average termination height
            if torch.mean(self.termination_height).item() > 0.65:
                self.cfg.rewards.regularization_scale *= (
                    1.0 + self.cfg.rewards.regularization_scale_gamma
                )
            elif torch.mean(self.termination_height).item() < 0.1:
                self.cfg.rewards.regularization_scale *= (
                    1.0 - self.cfg.rewards.regularization_scale_gamma
                )
            self.cfg.rewards.regularization_scale = max(
                min(
                    self.cfg.rewards.regularization_scale,
                    self.cfg.rewards.regularization_scale_range[1],
                ),
                self.cfg.rewards.regularization_scale_range[0],
            )
        elif self.cfg.rewards.regularization_scale_curriculum_type == "step_episode":
            # Step curriculum based on the average episode length
            if torch.mean(self.episode_length.float()).item() > 420.0:
                self.cfg.rewards.regularization_scale *= (
                    1.0 + self.cfg.rewards.regularization_scale_gamma
                )
            elif torch.mean(self.episode_length.float()).item() < 50.0:
                self.cfg.rewards.regularization_scale *= (
                    1.0 - self.cfg.rewards.regularization_scale_gamma
                )
            self.cfg.rewards.regularization_scale = max(
                min(
                    self.cfg.rewards.regularization_scale,
                    self.cfg.rewards.regularization_scale_range[1],
                ),
                self.cfg.rewards.regularization_scale_range[0],
            )

    def compute_observations(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0 * self.yaw, 0 * self.yaw, self.yaw)
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3 dims
                imu_obs,  # 2 dims
                self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                self.reindex(self.action_history_buf[:, -1]),
            ),
            dim=-1,
        )
        if self.cfg.noise.add_noise and self.headless:
            obs_buf += (
                (2 * torch.rand_like(obs_buf) - 1)
                * self.noise_scale_vec
                * min(
                    self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24), 1.0
                )
            )
        elif self.cfg.noise.add_noise and not self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        else:
            obs_buf += 0.0

        if self.cfg.domain_rand.domain_rand_general:
            priv_latent = torch.cat(
                (
                    self.mass_params_tensor,
                    self.friction_coeffs_tensor,
                    self.motor_strength[0] - 1,
                    self.motor_strength[1] - 1,
                    self.base_lin_vel,
                ),
                dim=-1,
            )
        else:
            priv_latent = torch.zeros(
                (self.num_envs, self.cfg.env.n_priv_latent), device=self.device
            )

        self.obs_buf = torch.cat(
            [obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1
        )

        if self.cfg.env.history_len > 0:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1),
            )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([self.contact_buf[:, 1:], self.contact_filt.float().unsqueeze(1)], dim=1),
        )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_z"][0],
            self.command_ranges["lin_vel_z"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :2] *= (
            torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip
        )

    # ================================================ Rewards ================================================== #
    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_base_lin_vel(self):
        return torch.norm(self.base_lin_vel, dim=-1)

    def _reward_ang_vel(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :3]), dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=-1)
    
    def _reward_torques(self):
        return torch.norm(self.torques, dim=-1)
    
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_torque_limits(self):
        out_of_limits = torch.sum((torch.abs(self.torques) / self.torque_limits - self.cfg.rewards.soft_torque_limit).clip(min=0), dim=1)
        return out_of_limits
    
    def _reward_energy(self):
        return torch.norm(torch.abs(self.torques * self.dof_vel), dim=-1)
    
    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        standing_flag = self.rigid_body_states[:, self.head_idx, 2] > 1.1
        dof_error[~standing_flag] *= 0
        return dof_error

    def _reward_feet_orientation(self):
        left_quat = self.rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1) **0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5

    def _reward_feet_height(self):
        feet_height = torch.mean(self.rigid_body_states[:, self.feet_indices, 2], dim=-1)
        return torch.exp(-10 * feet_height)

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_feet_distance_continuous(self):
        foot_pos = self.rigid_body_states[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.abs(foot_dist - fd)
        d_max = torch.abs(foot_dist - max_df)
        return (torch.exp(-d_min * 100) + torch.exp(-d_max * 100)) / 2.0

    def _reward_knee_distance_continuous(self):
        knee_pos = self.rigid_body_states[:, self.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        min_kd = self.cfg.rewards.min_dist
        max_kd = self.cfg.rewards.max_knee_dist
        d_min = torch.abs(knee_dist - min_kd)
        d_max = torch.abs(knee_dist - max_kd)
        return (torch.exp(-d_min * 100) + torch.exp(-d_max * 100)) / 2.0

    def _reward_base_roll_gravity_error_cosine(self):
        # Down
        # quat: [[ 0.7071  0.     -0.7071  0.    ]], gravity: [[9.809812   0.         0.00018883]]
        # quat: [[0.    0.707 0.    0.707]], gravity: [[9.807038  0.        0.0029622]]
        # UP
        # quat: [[ 0.    -0.707  0.     0.707]], gravity: [[-9.807038  -0.         0.0029622]]
        # quat: [[0.7071 0.     0.7071 0.    ]], gravity: [[-9.809812   -0.          0.00018883]]
        target_gravity = [-9.81, -0., -0.]
        target_gravity = torch.tensor(target_gravity, device=self.device, dtype=torch.float)
        gravity_error = 1 - torch.nn.functional.cosine_similarity(self.projected_gravity, target_gravity, dim=-1)  # [0, 2]
        return gravity_error

    def _reward_head_roll_gravity_error_cosine(self):
        target_gravity = [-9.81, -0., -0.]
        target_gravity = torch.tensor(target_gravity, device=self.device, dtype=torch.float)
        head_quat = self.rigid_body_rot[:, self.head_idx]
        head_projected_gravity = quat_rotate_inverse(head_quat, self.gravity_vec)
        gravity_error = 1 - torch.nn.functional.cosine_similarity(head_projected_gravity, target_gravity, dim=-1)
        return gravity_error

    def _reward_torso_roll_gravity_error_cosine(self):
        target_gravity = [-9.81, -0., -0.]
        target_gravity = torch.tensor(target_gravity, device=self.device, dtype=torch.float)
        torso_quat = self.rigid_body_rot[:, self.torso_idx]
        torso_projected_gravity = quat_rotate_inverse(torso_quat, self.gravity_vec)
        gravity_error = 1 - torch.nn.functional.cosine_similarity(torso_projected_gravity, target_gravity, dim=-1)
        return gravity_error

    def _reward_knee_roll_gravity_error_cosine(self):
        target_gravity = [-9.81, -0., -0.]
        target_gravity = torch.tensor(target_gravity, device=self.device, dtype=torch.float)
        left_knee_quat = self.rigid_body_rot[:, self.knee_indices[0]]
        left_knee_projected_gravity = quat_rotate_inverse(left_knee_quat, self.gravity_vec)
        right_knee_quat = self.rigid_body_rot[:, self.knee_indices[1]]
        right_knee_projected_gravity = quat_rotate_inverse(right_knee_quat, self.gravity_vec)
        gravity_error_left = 1 - torch.nn.functional.cosine_similarity(left_knee_projected_gravity, target_gravity, dim=-1)
        gravity_error_right = 1 - torch.nn.functional.cosine_similarity(right_knee_projected_gravity, target_gravity, dim=-1)
        return (gravity_error_left + gravity_error_right) / 2.0
