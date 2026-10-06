"""MimicGen / robomimic integration for DexCraft.

Importing this module registers the DexCraft MimicGen configs and env interface, and teaches robomimic
(which MimicGen uses to create environments) about ManiSkill environments.
"""
import robomimic.envs.env_base as EB
import robomimic.utils.env_utils as EnvUtils

import dexcraft.mimicgen.configs  # noqa: F401  (registers MG_Config classes)
import dexcraft.mimicgen.interface  # noqa: F401  (registers MG_DexCraft env interface)
from dexcraft.mimicgen.env_wrapper import MANISKILL_TYPE, EnvManiSkill


def _register_with_robomimic():
    if getattr(EnvUtils, "_dexcraft_registered", False):
        return
    EB.EnvType.MANISKILL_TYPE = MANISKILL_TYPE
    get_env_class = EnvUtils.get_env_class

    def get_env_class_with_maniskill(env_meta=None, env_type=None, env=None):
        env_type = EnvUtils.get_env_type(env_meta=env_meta, env_type=env_type, env=env)
        if env_type == MANISKILL_TYPE:
            return EnvManiSkill
        return get_env_class(env_meta=env_meta, env_type=env_type, env=env)

    EnvUtils.get_env_class = get_env_class_with_maniskill
    EnvUtils._dexcraft_registered = True


_register_with_robomimic()
