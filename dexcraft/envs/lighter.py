import numpy as np
from mani_skill.utils.registration import register_env

from dexcraft.envs.base import DexCraftEnv


@register_env("Lighter-v1", max_episode_steps=30000)
class LighterEnv(DexCraftEnv):
    """Pick up the lighter, move it to the goal pose and press the ignition lever."""

    obj_id = 100319
    obj_scale = 0.105
    obj_friction = (10.0, 1.0)
    obj_joint = dict(joint="hinge")
    obj_joint_drive = (0.0, 0.0)
    step_after_load_tool = True
    obj_joint_drive_target = 0.2
    obj_rot_bounds = ((0, np.pi / 3), (-np.pi / 3, 0), (-np.pi / 6, np.pi / 6))

    goal_rot_bounds = ((0, np.pi / 3), (-np.pi / 12, np.pi / 12), (-np.pi / 6, np.pi / 6))
    target_offset = ([0.1, 0, 0.1], ("y", -30))
    trigger_thresh = (">=", 0.5)
