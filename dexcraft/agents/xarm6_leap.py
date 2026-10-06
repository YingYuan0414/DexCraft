from copy import deepcopy

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation

from dexcraft.utils import ASSET_DIR


@register_agent()
class XArm6Leap(BaseAgent):
    """xArm6 arm with a LEAP hand.

    Action: 6-D arm command followed by 16 hand joint commands.
      - `pd_ee_target_delta_pose`: raw end-effector delta (m, rad), clipped to [-0.1, 0.1]. Used by MimicGen
        data generation and policies.
      - `pd_ee_target_delta_pose_normalized`: same controller, but the arm action in [-1, 1] is scaled to
        [-0.1, 0.1]. Used for keyboard teleoperation / source demos (MimicGen's env interface assumes it).
    Hand delta actions are always normalized: [-1, 1] -> [-0.1, 0.1] rad.
    """

    uid = "xarm6_leap"
    urdf_path = str(ASSET_DIR / "robot" / "xarm6_description" / "xarm6_leap.urdf")
    disable_self_collisions = True  # overridden per task, see DexCraftEnv.agent_disable_self_collisions

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([-0.014, -0.673, -0.038, -0.079, 0.623, -0.018] + [0] * 16),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(qpos=np.zeros(22), pose=sapien.Pose([0, 0, 0])),
    )

    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    # ordering matches the URDF
    hand_joint_names = [
        "leap_1", "leap_5", "leap_9", "leap_12",
        "leap_0", "leap_4", "leap_8", "leap_13",
        "leap_2", "leap_6", "leap_10", "leap_14",
        "leap_3", "leap_7", "leap_11", "leap_15",
    ]

    arm_stiffness = 100
    arm_damping = 10
    arm_friction = [0.1] * 6
    arm_force_limit = 100

    hand_stiffness = 3
    hand_damping = 0.1
    hand_friction = 5
    hand_force_limit = 25

    ee_link_name = "palm_lower"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        links = self.robot.get_links()
        get = lambda name: sapien_utils.get_obj_by_name(links, name)
        self.tcp = get(self.ee_link_name)
        self.index_finger_links = [get(n) for n in ["mcp_joint", "pip", "dip", "fingertip"]]
        self.mid_finger_links = [get(n) for n in ["mcp_joint_2", "pip_2", "dip_2", "fingertip_2"]]
        self.little_finger_links = [get(n) for n in ["mcp_joint_3", "pip_3", "dip_3", "fingertip_3"]]
        self.thumb_links = [get(n) for n in ["pip_4", "thumb_dip", "thumb_fingertip"]]
        self.all_hand_links = (
            [self.tcp] + self.index_finger_links + self.mid_finger_links + self.little_finger_links + self.thumb_links
        )

    @property
    def _controller_configs(self):
        # ---------------- arm ---------------- #
        arm_kwargs = dict(
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, friction=self.arm_friction,
        )
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names, lower=None, upper=None, normalize_action=False, **arm_kwargs
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names, lower=-0.1, upper=0.1, use_delta=True, **arm_kwargs
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names, pos_lower=-0.1, pos_upper=0.1,
            ee_link=self.ee_link_name, urdf_path=self.urdf_path, **arm_kwargs,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names, pos_lower=-0.1, pos_upper=0.1, rot_lower=-0.1, rot_upper=0.1,
            ee_link=self.ee_link_name, urdf_path=self.urdf_path, normalize_action=False, **arm_kwargs,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names, pos_lower=None, pos_upper=None,
            ee_link=self.ee_link_name, urdf_path=self.urdf_path, use_delta=False, normalize_action=False,
            **arm_kwargs,
        )
        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True
        arm_pd_ee_target_delta_pose_normalized = deepcopy(arm_pd_ee_target_delta_pose)
        arm_pd_ee_target_delta_pose_normalized.normalize_action = True

        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names, -1.0, 1.0, self.arm_damping, self.arm_force_limit, self.arm_friction
        )
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names, None, None, self.arm_stiffness, self.arm_damping,
            self.arm_force_limit, self.arm_friction, normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names, -0.1, 0.1, self.arm_stiffness, self.arm_damping,
            self.arm_force_limit, friction=self.arm_friction, use_delta=True,
        )

        # ---------------- hand ---------------- #
        hand_args = (self.hand_stiffness, self.hand_damping, self.hand_force_limit, self.hand_friction)
        hand_joint_pos = PDJointPosControllerConfig(self.hand_joint_names, None, None, *hand_args, normalize_action=False)
        hand_joint_delta_pos = PDJointPosControllerConfig(self.hand_joint_names, -0.1, 0.1, *hand_args, use_delta=True)
        hand_joint_target_delta_pos = deepcopy(hand_joint_delta_pos)
        hand_joint_target_delta_pos.use_target = True
        hand_joint_target_pos = deepcopy(hand_joint_pos)
        hand_joint_target_pos.use_target = True

        hand = hand_joint_target_delta_pos
        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, hand=hand),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, hand=hand),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, hand=hand),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, hand=hand),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, hand=hand),
            pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos, hand=hand),
            pd_ee_target_delta_pos=dict(arm=arm_pd_ee_target_delta_pos, hand=hand),
            # MimicGen data generation
            pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose, hand=hand),
            # keyboard teleoperation (source demos)
            pd_ee_target_delta_pose_normalized=dict(arm=arm_pd_ee_target_delta_pose_normalized, hand=hand),
            # delta end-effector pose, absolute hand joint positions
            pd_ee_target_delta_pose_hand_pose=dict(arm=arm_pd_ee_target_delta_pose, hand=hand_joint_target_pos),
            pd_joint_vel=dict(arm=arm_pd_joint_vel, hand=hand),
            pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel, hand=hand),
            pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel, hand=hand),
        )
        return deepcopy_dict(controller_configs)

    def is_grasping(self, object: Actor, min_force=0.5):
        """Middle and little fingertips both in contact with the object (any link if articulated)."""
        object_list = object.get_links() if isinstance(object, Articulation) else [object]
        mid = little = False
        for obj in object_list:
            mid_force = torch.linalg.norm(self.scene.get_pairwise_contact_forces(self.mid_finger_links[-1], obj), axis=1)
            little_force = torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(self.little_finger_links[-1], obj), axis=1
            )
            mid = (mid_force >= min_force) | mid
            little = (little_force >= min_force) | little
        return mid & little

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
