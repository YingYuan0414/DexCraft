"""MimicGen config classes for the DexCraft tasks (registered by name, e.g. `spray_bottle`).

The values below are defaults; the configs actually used are the JSON files in configs/mimicgen/.
"""
from mimicgen.configs.config import MG_Config


class DexCraftConfig(MG_Config):
    NAME = "dexcraft"
    TYPE = "maniskill"

    def task_config(self):
        subtask = dict(
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0001,
            num_interpolation_steps=50,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # 1) grasp the tool, 2) move it to the goal pose, 3) actuate it
        self.task.task_spec.subtask_1 = dict(
            object_ref="object", subtask_term_signal="grasp", subtask_term_offset_range=(5, 10), **subtask
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="goal", subtask_term_signal="place", subtask_term_offset_range=(5, 10), **subtask
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="goal", subtask_term_signal=None, subtask_term_offset_range=None, **subtask
        )
        self.task.task_spec.do_not_lock_keys()


class SprayBottleConfig(DexCraftConfig):
    NAME = "spray_bottle"


class LighterConfig(DexCraftConfig):
    NAME = "lighter"


class DispenserConfig(DexCraftConfig):
    NAME = "dispenser"


class PenConfig(DexCraftConfig):
    NAME = "pen"


class PliersConfig(DexCraftConfig):
    NAME = "pliers"


class StaplerConfig(DexCraftConfig):
    NAME = "stapler"
