# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

HI2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/hi2/usd/hi2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={  
            "left_hip_roll_joint": 0.01,
            "left_hip_pitch_joint": -0.15,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.31,
            "left_ankle_pitch_joint": -0.12,
            "left_ankle_roll_joint": -0.0,
            "right_hip_roll_joint": -0.01,
            "right_hip_pitch_joint": -0.15,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.31,
            "right_ankle_pitch_joint": -0.12,
            "right_ankle_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.1,
            "left_shoulder_roll_joint": 0.15,
            "left_shoulder_yaw_joint": -0.0,
            "left_elbow_joint": -0.2,
            "right_shoulder_pitch_joint": 0.1,
            "right_shoulder_roll_joint": -0.15,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.2,   

            # "left_hip_roll_joint": 0.0,
            # "left_hip_pitch_joint": -0.5,
            # "left_hip_yaw_joint": 0.0,
            # "left_knee_joint": 1.0,
            # "left_ankle_pitch_joint": -0.5,
            # "left_ankle_roll_joint": -0.0,
            # "right_hip_roll_joint": -0.0,
            # "right_hip_pitch_joint": -0.5,
            # "right_hip_yaw_joint": 0.0,
            # "right_knee_joint": 1.0,
            # "right_ankle_pitch_joint": -0.5,
            # "right_ankle_roll_joint": 0.0,
            # "left_shoulder_pitch_joint": 0.0,
            # "left_shoulder_roll_joint": 0.1,
            # "left_shoulder_yaw_joint": -0.0,
            # "left_elbow_joint": -0.3,
            # "right_shoulder_pitch_joint": 0.0,
            # "right_shoulder_roll_joint": -0.1,
            # "right_shoulder_yaw_joint": 0.0,
            # "right_elbow_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "(left|right)_hip_roll_joint",
                "(left|right)_hip_pitch_joint",
                "(left|right)_hip_yaw_joint",
                "(left|right)_knee_joint",
            ],
            effort_limit_sim={
                "(left|right)_hip_roll_joint": 180,
                "(left|right)_hip_pitch_joint": 300,
                "(left|right)_hip_yaw_joint": 180,
                "(left|right)_knee_joint": 300,
            },
            velocity_limit_sim={
                "(left|right)_hip_roll_joint": 15.6,
                "(left|right)_hip_pitch_joint": 15.6,
                "(left|right)_hip_yaw_joint": 15.6,
                "(left|right)_knee_joint": 15.6,
            },
            stiffness={
                "(left|right)_hip_roll_joint": 700,
                "(left|right)_hip_pitch_joint": 700,
                "(left|right)_hip_yaw_joint": 500,
                "(left|right)_knee_joint": 700,
            },
            damping={
                "(left|right)_hip_roll_joint": 10,
                "(left|right)_hip_pitch_joint": 10,
                "(left|right)_hip_yaw_joint": 5,
                "(left|right)_knee_joint": 10,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "(left|right)_ankle_pitch_joint",
                "(left|right)_ankle_roll_joint",
            ],
            effort_limit_sim={
                "(left|right)_ankle_pitch_joint": 60,
                "(left|right)_ankle_roll_joint": 30,
            },
            velocity_limit_sim={
                "(left|right)_ankle_pitch_joint": 12.8,
                "(left|right)_ankle_roll_joint": 7.8,
            },
            stiffness={
                "(left|right)_ankle_pitch_joint": 30,
                "(left|right)_ankle_roll_joint": 16.8,
            },
            damping={
                "(left|right)_ankle_pitch_joint": 2.5,
                "(left|right)_ankle_roll_joint": 1.4,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "(left|right)_shoulder_pitch_joint",
                "(left|right)_shoulder_roll_joint",
                "(left|right)_shoulder_yaw_joint",
                "(left|right)_elbow_joint",
            ],
            effort_limit_sim={
                "(left|right)_shoulder_pitch_joint": 52.5,
                "(left|right)_shoulder_roll_joint": 52.5,
                "(left|right)_shoulder_yaw_joint": 52.5,
                "(left|right)_elbow_joint": 52.5,
            },
            velocity_limit_sim={
                "(left|right)_shoulder_pitch_joint": 14.1,
                "(left|right)_shoulder_roll_joint": 14.1,
                "(left|right)_shoulder_yaw_joint": 14.1,
                "(left|right)_elbow_joint": 14.1,
            },
            stiffness={
                "(left|right)_shoulder_pitch_joint": 60,
                "(left|right)_shoulder_roll_joint": 20,
                "(left|right)_shoulder_yaw_joint": 10,
                "(left|right)_elbow_joint": 10,
            },
            damping={
                "(left|right)_shoulder_pitch_joint": 3,
                "(left|right)_shoulder_roll_joint": 1.5,
                "(left|right)_shoulder_yaw_joint": 1,
                "(left|right)_elbow_joint": 1,
            },
        ),
    },
)
