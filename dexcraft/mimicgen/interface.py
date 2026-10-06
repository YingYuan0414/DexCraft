"""MimicGen environment interface for the DexCraft tasks.

NOTE: actions are interpreted in the units of the teleoperation control mode
(`pd_ee_target_delta_pose_normalized`), in which source demos are recorded. `target_pose_to_action` returns
actions for the un-normalized `pd_ee_target_delta_pose` mode used during generation, which is why the
generation configs set `env_meta_update_kwargs.env_kwargs.control_mode = "pd_ee_target_delta_pose"`.
"""
import mimicgen.utils.pose_utils as PoseUtils
import numpy as np
import robosuite.utils.transform_utils as T
import torch
from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles
from mimicgen.env_interfaces.base import MG_EnvInterface
from scipy.spatial.transform import Rotation as R

MAX_DPOS = 0.1
MAX_DROT = np.pi / 18


def clip_and_scale_action(action, low, high):
    """Clip action to [-1, 1] and scale according to a range [low, high]."""
    action = torch.clip(action, -1, 1)
    return 0.5 * (high + low) + 0.5 * (high - low) * action


def clip_rotation(delta_rot_mat, max_angle=MAX_DROT):
    rotvec = R.from_matrix(delta_rot_mat).as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    return R.from_rotvec(axis * min(angle, max_angle)).as_matrix()


class ManiSkillInterface(MG_EnvInterface):
    INTERFACE_TYPE = "maniskill"

    @property
    def base_env(self):
        return self.env.unwrapped

    def get_robot_eef_pose(self):
        return self.base_env.agent.tcp_pose.to_transformation_matrix().squeeze().numpy()

    def target_pose_to_action(self, target_pose, relative=True):
        """Convert a 4x4 target eef pose to the arm part of an action."""
        curr_pose = self.get_robot_eef_pose()
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        if relative:
            delta_position = np.clip((target_pos - curr_pos) / MAX_DPOS, -1.0, 1.0) * MAX_DPOS
            delta_position = clip_and_scale_action(torch.tensor(delta_position), -MAX_DPOS, MAX_DPOS).numpy()
            delta_rot_mat = clip_rotation(target_rot @ curr_rot.T)
            delta_rotation = matrix_to_euler_angles(torch.tensor(delta_rot_mat), "XYZ").numpy()
            return np.concatenate([delta_position, delta_rotation])

        abs_rotation = T.quat2axisangle(T.mat2quat(target_rot))
        return np.concatenate([target_pos, abs_rotation])

    def action_to_target_pose(self, action, relative=True):
        """Convert the arm part of an action to a 4x4 target eef pose."""
        if not relative:
            target_pos = action[:3]
            target_rot = T.quat2mat(T.axisangle2quat(action[3:6]))
        else:
            curr_pos, curr_rot = PoseUtils.unmake_pose(self.get_robot_eef_pose())
            target_pos = curr_pos + action[:3]
            delta_rot_mat = euler_angles_to_matrix(torch.tensor(action[3:6]), "XYZ").numpy()
            target_rot = np.matmul(delta_rot_mat, curr_rot)
        return PoseUtils.make_pose(target_pos, target_rot)

    def action_to_gripper_action(self, action):
        """The hand joint commands play the role of the gripper action."""
        return action[6:]


class MG_DexCraft(ManiSkillInterface):
    """Shared by all DexCraft tasks: subtasks are (1) grasp the tool, (2) move it to the goal, (3) actuate it."""

    def get_object_poses(self):
        return dict(
            object=self.base_env.instance.pose.to_transformation_matrix().squeeze().numpy(),
            goal=self.base_env.goal_site.pose.to_transformation_matrix().squeeze().numpy(),
        )

    def get_subtask_term_signals(self):
        info = self.base_env.evaluate()
        return dict(
            grasp=int(info["is_grasped"]),
            place=int(info["is_obj_placed"] and info["is_obj_oriented"]),
        )
