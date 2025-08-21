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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .base.humanoid import Humanoid

# G1 with waist dof
from .g1waist.g1waist_up_config import G1WaistHumanUPCfg, G1WaistHumanUPCfgPPO
from .g1waist.g1waist_up import G1WaistHumanUP
from .g1waistroll.g1waistroll_up_config import G1WaistRollHumanUPCfg, G1WaistRollHumanUPCfgPPO
from .g1waistroll.g1waistroll_up import G1WaistRollHumanUP

from .g1track.g1waist_track_config import G1WaistTrackCfg, G1WaistTrackCfgPPO
from .g1track.g1waist_track import G1WaistTrack

from .g1rolltrack.g1waistroll_track_config import G1WaistRollTrackCfg, G1WaistRollTrackCfgPPO
from .g1rolltrack.g1waistroll_track import G1WaistRollTrack

from .go2.go2_up import Go2UP
from .go2.go2_up_config import Go2UPCfg, Go2UPCfgPPO
from legged_gym.gym_utils.task_registry import task_registry

# ======================= environment registration =======================

task_registry.register("g1waist_up", G1WaistHumanUP, G1WaistHumanUPCfg(), G1WaistHumanUPCfgPPO())

task_registry.register("g1waist_track", G1WaistTrack, G1WaistTrackCfg(), G1WaistTrackCfgPPO())

task_registry.register("g1waistroll_up", G1WaistRollHumanUP, G1WaistRollHumanUPCfg(), G1WaistRollHumanUPCfgPPO())

task_registry.register("g1waistroll_track", G1WaistRollTrack, G1WaistRollTrackCfg(), G1WaistRollTrackCfgPPO())

task_registry.register("go2up", Go2UP, Go2UPCfg(), Go2UPCfgPPO())