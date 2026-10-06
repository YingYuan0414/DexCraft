import sapien
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from dexcraft.envs.base import DexCraftEnv


def build_box(scene: ManiSkillScene, half_size):
    builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial(base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5)
    builder.add_box_collision(half_size=half_size)
    builder.add_box_visual(half_size=half_size, material=mat)
    return builder


@register_env("Pliers-v1", max_episode_steps=30000)
class PliersEnv(DexCraftEnv):
    """Pick up the pliers resting on a block, move them to the goal pose and close them."""

    obj_id = 100188
    obj_scale = 0.15
    obj_friction = (8.0, 0.8)
    obj_joint = dict(joint="hinge")
    obj_joint_drive_target = 0.5

    target_offset = ([-0.12, 0, 0], ("x", 90))
    trigger_thresh = ("<=", 0.05)

    def _load_extra_actors(self):
        builder = build_box(self.scene, half_size=(0.04, 0.04, 0.03))
        builder.set_initial_pose(sapien.Pose([0, 0, 0.015]))
        self.base = builder.build(name="base")

    def _initialize_extra_actors(self, obj_xyz, obj_quat):
        base_xyz = obj_xyz.clone()
        base_xyz[:, -1] = 0.015
        base_xyz[:, 0] += 0.03
        self.base.set_pose(Pose.create_from_pq(base_xyz, obj_quat))
