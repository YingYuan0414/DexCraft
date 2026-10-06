from mani_skill.utils.registration import register_env

from dexcraft.envs.base import DexCraftEnv


@register_env("Dispenser-v1", max_episode_steps=30000)
class DispenserEnv(DexCraftEnv):
    """Pick up the soap dispenser, move it to the goal pose and press the pump."""

    obj_id = 101501
    obj_scale = 0.12
    obj_joint = dict(joint="slider", name="pressing_lid")
    obj_joint_drive = (200.0, 50.0)
    obj_joint_drive_target = 0.0

    target_offset = ([-0.08, 0, 0.12], ("y", 15))
    trigger_thresh = (">=", 0.015)
