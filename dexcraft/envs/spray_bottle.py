from mani_skill.utils.registration import register_env

from dexcraft.envs.base import DexCraftEnv


@register_env("SprayBottle-v1", max_episode_steps=30000)
class SprayBottleEnv(DexCraftEnv):
    """Pick up the spray bottle, aim it at the goal pose and squeeze the trigger."""

    agent_disable_self_collisions = False

    obj_id = 101463
    obj_scale = 0.13
    obj_joint = dict(joint="hinge")
    obj_joint_drive_target = 0.2

    target_offset = ([-0.08, 0, 0.12], ("y", 15))
    trigger_thresh = (">=", 0.3)
