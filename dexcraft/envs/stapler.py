import numpy as np
from mani_skill.utils.registration import register_env

from dexcraft.envs.base import DexCraftEnv


@register_env("Stapler-v1", max_episode_steps=30000)
class StaplerEnv(DexCraftEnv):
    """Pick up the stapler lying on its side, move it to the goal pose and press it shut."""

    obj_id = 103280
    obj_scale = 0.105
    obj_half_length = 0.025
    obj_friction = (10.0, 1.0)
    obj_joint = dict(joint="hinge")
    step_after_load_tool = True
    obj_joint_drive_target = 0.9
    obj_rot_lock = (False, True)
    obj_rot_bounds = ((np.pi / 2, np.pi / 2 + 0.001), (-np.pi / 2, -np.pi / 2 + 0.001), (5 * np.pi / 6, 7 * np.pi / 6))

    goal_rot_bounds = ((0, np.pi / 3), (-np.pi / 12, np.pi / 12), (5 * np.pi / 6, 7 * np.pi / 6))
    target_offset = ([-0.08, 0, 0], ("y", 90))
    trigger_thresh = ("<=", 0.05)
