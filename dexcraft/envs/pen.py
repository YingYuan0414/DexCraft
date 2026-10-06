import sapien
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from dexcraft.envs.base import DexCraftEnv


def build_stand(scene: ManiSkillScene, inner_radius, outer_radius, depth):
    """Square tube that holds the pen upright."""
    builder = scene.create_actor_builder()
    half_thickness = (outer_radius - inner_radius) * 0.5
    offset = half_thickness + inner_radius
    half_sizes = [
        [half_thickness, outer_radius, depth],
        [half_thickness, outer_radius, depth],
        [outer_radius, half_thickness, depth],
        [outer_radius, half_thickness, depth],
    ]
    poses = [
        sapien.Pose([offset, 0, 0]),
        sapien.Pose([-offset, 0, 0]),
        sapien.Pose([0, offset, 0]),
        sapien.Pose([0, -offset, 0]),
    ]
    mat = sapien.render.RenderMaterial(base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5)
    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder


@register_env("Pen-v1", max_episode_steps=30000)
class PenEnv(DexCraftEnv):
    """Take the pen out of its stand, move it to the goal pose and click the button."""

    obj_id = 102922
    obj_scale = 0.15
    obj_half_length = 0.16
    obj_friction = (8.0, 0.8)
    obj_joint = dict(joint="slider")
    obj_joint_drive_target = -0.1

    goal_min_height = 0.2
    target_offset = ([0, 0, -0.16], ("y", 90))
    trigger_thresh = (">=", 0.005)

    def _load_extra_actors(self):
        builder = build_stand(self.scene, 0.02, 0.06, 0.05)
        builder.set_initial_pose(sapien.Pose([0, 0, 0.025]))
        self.base = builder.build(name="base")

    def _initialize_extra_actors(self, obj_xyz, obj_quat):
        base_xyz = obj_xyz.clone()
        base_xyz[:, -1] = 0.05
        self.base.set_pose(Pose.create_from_pq(base_xyz, obj_quat))
