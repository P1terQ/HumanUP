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

from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO


class Go2UPCfg(HumanoidCfg):
    class env(HumanoidCfg.env):
        num_envs = 4096
        num_actions = 12#23  # NOTE: the wrist dof is removed
        n_priv = 0
        n_proprio = 3 + 2 + 3 * num_actions  # NOTE
        n_priv_latent = 4 + 1 + 2 * num_actions + 3
        history_len = 10

        num_observations = n_proprio + n_priv_latent + history_len * n_proprio + n_priv

        num_privileged_obs = None

        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 10

        randomize_start_pos = False
        randomize_start_yaw = False

        history_encoding = True
        contact_buf_len = 10

        normalize_obs = False#True

        terminate_on_velocity = False#True
        terminate_on_height = False#True

        no_symmetry_after_stand = True

    class terrain(HumanoidCfg.terrain):
        mesh_type = "plane"

    class init_state(HumanoidCfg.init_state):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    class control(HumanoidCfg.control):
        # stiffness = {
        #     "hip_yaw": 150,
        #     "hip_roll": 150,
        #     "hip_pitch": 200,
        #     "knee": 200,
        #     "ankle": 20,
        #     "shoulder": 40,
        #     "elbow": 40,
        #     "waist": 200,
        # }  # [N*m/rad]
        # damping = {
        #     "hip_yaw": 5,
        #     "hip_roll": 5,
        #     "hip_pitch": 5,
        #     "knee": 5,
        #     "ankle": 4,
        #     "shoulder": 10,
        #     "elbow": 10,
        #     "waist": 5,
        # }  # [N*m/rad]  # [N*m*s/rad]
        stiffness = {'joint': 40.0}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        action_scale = 0.5
        decimation = 20

    class sim(HumanoidCfg.sim):
        dt = 0.001  # NOTE
        gravity = [0, 0, -9.81]

    class normalization(HumanoidCfg.normalization):
        clip_actions = 5

    class asset(HumanoidCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        flip_visual_attachments = True
        # for both joint and link name
        torso_name: str = "base"  # humanoid pelvis part
        chest_name: str = "base"  # humanoid chest part
        forehead_name: str = "base"  # humanoid head part

        waist_name: str = "base"

        # for link name
        thigh_name: str = "thigh"
        shank_name: str = "calf"
        foot_name: str = "foot"  # foot_pitch is not used
        upper_arm_name: str = "hip"
        lower_arm_name: str = "calf"
        hand_name: str = "head"

        # for joint name
        hip_name: str = "hip"
        hip_roll_name: str = "hip"
        hip_yaw_name: str = "hip"
        hip_pitch_name: str = "hip"
        knee_name: str = "thigh"
        ankle_name: str = "calf"
        ankle_pitch_name: str = "calf"
        shoulder_name: str = "calf"
        shoulder_pitch_name: str = "calf"
        shoulder_roll_name: str = "calf"
        shoulder_yaw_name: str = "calf"
        elbow_name: str = "calf"

        feet_bodies = ["foot"]
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["shoulder", "elbow", "hip"]  # NOTE: now there is no penalization on contacts
        terminate_after_contacts_on = []  # NOTE: now there is no termination after contacts
        dof_armature =[0.0,0.0,0.0,0.0,0.0,0.001]*2 + [0.0]* 3 + [0.0]*8

    class rewards(HumanoidCfg.rewards):
        regularization_names = [
            "dof_error",
            "dof_error_upper",
            "dof_vel",
            "feet_stumble",
            "feet_contact_forces",
            "feet_height",
            "feet_height_target_error",
            "feet_height_target_error_exp",
            "stand_on_feet",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "dof_pos_limits",
            "dof_torque_limits",
            "collision",
            "torque_penalty",
        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8, 2.0]
        regularization_scale_curriculum = True
        regularization_scale_curriculum_type = "step_height"  # ["sin", "step_height", "step_episode"]
        regularization_scale_gamma = 0.0001
        regularization_scale_curriculum_iterations = 100000

        standing_scale = 1.0
        standing_scale_range = [0.1, 0.2]
        standing_scale_curriculum = True
        standing_scale_curriculum_type = "cos"
        standing_scale_gamma = 0.0001
        standing_scale_curriculum_iterations = 50000

        class scales:
            base_height_exp = 5
            head_height_exp = 5
            delta_base_height = 1
            feet_contact_forces_increase = 1
            stand_on_feet = 2.5
            orientation = -1
            body_up_exp = 0.25
            feet_height = 2.5
            feet_distance = 2
            soft_symmetry_body = 0#-1
            soft_symmetry_waist = 0#-1
            feet_orientation = -0.5
            foot_slip = -1


            termination = -500

            # smooth reward
            dof_error = -0.03
            base_lin_vel = -0.1
            ang_vel = -0.1
            dof_vel = -0.0001
            action_rate = -0.1
            torques = -0.000001
            dof_pos_limits = -5
            dof_torque_limits = -0.1
            energy = -0.0001
            dof_acc = -0.0000001


        base_height_target = 0.728  # NOTE: the target height of the base
        head_height_target = 1.3  # NOTE: the target height of the head
        target_feet_height = 0.1  # NOTE: the target height of the feet
        min_dist = 0.25
        max_dist = 1.0
        max_knee_dist = 0.5
        target_joint_pos_scale = 0.17
        cycle_time = 0.64
        double_support_threshold = 0.1
        only_positive_rewards = False
        clip_inf_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        clip_inf_rewards = False
        max_contact_force = 500
        max_contact_force_head = 250  # NOTE: the max contact force for head
        max_contact_force_torso = 250  # NOTE: the max contact force for torso
        termination_height = 0.0

    class domain_rand:
        # TODO: try using the force compensation to drag up
        drag_robot_up = False  # True: robot is dragged up
        drag_robot_by_force = True  # True: robot is dragged up by force
        drag_robot_part = "head"  # the part of the robot to be dragged up, choose from ["head", "torso", "base"]
        drag_force = 1500  # drag force [N] (1000N ~ 100kg)
        drag_force_curriculum = True
        drag_force_curriculum_type = "sin"  # ["sin", "linear"]
        drag_force_curriculum_target_height = 0.728
        drag_interval = 50  # drag robot up every this many steps
        drag_when_falling = False  # True: drag robot up when falling； False: drag robot up regularly every drag_interval steps
        force_compenstation = False  # True: force compensation is applied
        min_drag_vel = 0.1  # min drag velocity [m/s]
        max_drag_vel = 0.5  # max drag velocity [m/s]

        domain_rand_general = False  # manually set this, setting from parser does not work;

        randomize_gravity = True and domain_rand_general
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)

        randomize_friction = True and domain_rand_general
        friction_range = [0.6, 2.0]

        randomize_base_mass = True and domain_rand_general
        added_mass_range = [-3.0, 3]

        randomize_base_com = True and domain_rand_general
        added_com_range = [-0.05, 0.05]

        push_robots = True and domain_rand_general
        push_interval_s = 4
        max_push_vel_xy = 1.0

        randomize_motor = True and domain_rand_general
        motor_strength_range = [0.8, 1.2]

        action_delay = True and domain_rand_general
        action_buf_len = 8

    class noise(HumanoidCfg.noise):
        add_noise = False
        noise_increasing_steps = 5000

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            imu = 0.05

    class commands:
        curriculum = False
        num_commands = 1
        resampling_time = 3.0  # time before command are changed[s]

        ang_vel_clip = 0.1
        lin_vel_clip = 0.1

        class ranges:
            lin_vel_x = [0.0, 0.0]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]
            lin_vel_z = [0.0, 0.3]
            ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]


class Go2UPCfgPPO(HumanoidCfgPPO):
    seed = 1

    class runner(HumanoidCfgPPO.runner):
        policy_class_name = "ActorCriticRMA"
        algorithm_class_name = "PPORMA"
        
        runner_class_name = "OnPolicyRunner"
        max_iterations = 50001  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "test"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

    class policy(HumanoidCfgPPO.policy):
       # action_std = [0.3, 0.3, 0.3, 0.4, 0.2, 0.2] * 2# + [0.1] * 3 + [0.2] * 8  # NOTE: the wrist dof is removed
        init_noise_std = 1.0

    class algorithm(HumanoidCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]  # NOTE: no grad penalty