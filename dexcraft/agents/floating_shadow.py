from copy import deepcopy

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.actor import Actor

from dexcraft.utils import ASSET_DIR


@register_agent()
class FloatingShadow(BaseAgent):
    """Floating Shadow hand (6 virtual root joints + 22 hand joints), used to replay Dexonomy grasps."""

    uid = "floating_shadow"
    urdf_path = str(ASSET_DIR / "robot" / "shadow_hand" / "shadow_hand_right_float.urdf")
    srdf_path = str(ASSET_DIR / "robot" / "shadow_hand" / "shadow_hand_right_float.srdf")
    disable_self_collisions = True

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([-0.615, 0, 0.1, np.pi / 2, 0, np.pi / 2] + [0] * 22),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    arm_joint_names = [
        "root_x_axis_joint", "root_y_axis_joint", "root_z_axis_joint",
        "root_x_rot_joint", "root_y_rot_joint", "root_z_rot_joint",
    ]
    hand_joint_names = [
        "FFJ4", "FFJ3", "FFJ2", "FFJ1",
        "MFJ4", "MFJ3", "MFJ2", "MFJ1",
        "RFJ4", "RFJ3", "RFJ2", "RFJ1",
        "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
        "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
    ]

    arm_stiffness = 100
    arm_damping = 10
    arm_friction = 0.1
    arm_force_limit = 100

    hand_stiffness = 3
    hand_damping = 0.1
    hand_friction = 5
    hand_force_limit = 25

    ee_link_name = "palm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        links = self.robot.get_links()
        get = lambda name: sapien_utils.get_obj_by_name(links, name)
        self.tcp = get(self.ee_link_name)
        self.index_finger_links = [get(n) for n in ["ffknuckle", "ffproximal", "ffmiddle", "ffdistal"]]
        self.mid_finger_links = [get(n) for n in ["mfknuckle", "mfproximal", "mfmiddle", "mfdistal"]]
        self.ring_finger_links = [get(n) for n in ["rfknuckle", "rfproximal", "rfmiddle", "rfdistal"]]
        self.little_finger_links = [get(n) for n in ["lfknuckle", "lfproximal", "lfmiddle", "lfdistal"]]
        self.thumb_links = [get(n) for n in ["thproximal", "thmiddle", "thdistal"]]
        # hand keypoints used for the `hand_pcd` observation (palm, index, middle, ring, thumb)
        self.all_hand_links = (
            [self.tcp] + self.index_finger_links + self.mid_finger_links + self.ring_finger_links + self.thumb_links
        )

    @property
    def _controller_configs(self):
        arm_kwargs = dict(
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, friction=self.arm_friction,
        )
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names, lower=None, upper=None, normalize_action=False, **arm_kwargs
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names, lower=-0.1, upper=0.1, use_delta=True, normalize_action=False, **arm_kwargs
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        hand_args = (self.hand_stiffness, self.hand_damping, self.hand_force_limit, self.hand_friction)
        hand_joint_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names, -0.01, 0.01, *hand_args, use_delta=True, normalize_action=False
        )
        hand_joint_target_delta_pos = deepcopy(hand_joint_delta_pos)
        hand_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, hand=hand_joint_target_delta_pos),
            # NOTE: the hand is always controlled with target-delta joint positions
            pd_joint_pos=dict(arm=arm_pd_joint_pos, hand=hand_joint_target_delta_pos),
            pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos, hand=hand_joint_target_delta_pos),
        )
        return deepcopy_dict(controller_configs)

    def is_grasping(self, object: Actor, min_force=0.01):
        """At least two fingers touch the object."""
        n_contacts = 0
        for finger in [self.index_finger_links, self.mid_finger_links, self.ring_finger_links,
                       self.little_finger_links, self.thumb_links]:
            in_contact = False
            for link in finger:
                force = torch.linalg.norm(self.scene.get_pairwise_contact_forces(link, object), axis=1)
                in_contact = (force >= min_force) | in_contact
            n_contacts += int(in_contact)
        return n_contacts >= 2

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
