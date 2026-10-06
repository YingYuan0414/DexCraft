"""Floating Shadow hand grasping a single object, used to turn Dexonomy grasps into trajectories."""
import os
from typing import Any, Dict, Optional

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from pytorch3d.ops import sample_farthest_points

import dexcraft.agents  # noqa: F401  (registers robots)

NUM_POINTS = 1024
SEG_OBJ_MIN = 38  # segmentation ids >= 38 belong to the object, the rest (after cropping the table) to the hand


@register_env("ShadowGrasp-v1", max_episode_steps=30000)
class ShadowGraspEnv(BaseEnv):
    """
    Args:
        obj_info: dict with
            obj_path: Dexonomy object folder (containing mesh/simplified.obj)
            obj_scale: mesh scale
            obj_pose: [x, y, z, qw, qx, qy, qz] object pose; the object is placed 0.2 m above it
    """

    SUPPORTED_ROBOTS = ["floating_shadow"]
    sensor_cam_eye_pos = [[0.3, 0, 0.6], [0, 0.3, 0.6]]
    sensor_cam_target_pos = [-0.1, 0, 0.1]
    human_cam_eye_pos = [0.6, 0.7, 0.6]
    human_cam_target_pos = [0.0, 0.0, 0.35]

    def __init__(self, *args, robot_uids="floating_shadow", robot_init_qpos_noise=0.02, obj_info=None, **kwargs):
        if robot_uids not in self.SUPPORTED_ROBOTS:
            raise NotImplementedError(f"{robot_uids} is not supported, choose from {self.SUPPORTED_ROBOTS}")
        assert obj_info is not None, "ShadowGrasp-v1 requires `obj_info`"
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.obj_info = obj_info
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        configs = []
        for i, eye_pos in enumerate(self.sensor_cam_eye_pos):
            pose = sapien_utils.look_at(eye=eye_pos, target=self.sensor_cam_target_pos)
            configs.append(CameraConfig(f"sensor_camera_{i}", pose, 128, 128, np.pi / 2, 0.01, 100))
        return configs

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=self.human_cam_eye_pos, target=self.human_cam_target_pos)
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # free-floating object without gravity, so it stays in place until grasped
        builder = self.scene.create_actor_builder()
        pose = self.obj_info["obj_pose"]
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 0.2] + pose[:3], q=pose[3:]))
        mesh_path = os.path.join(self.obj_info["obj_path"], "mesh/simplified.obj")
        scale = self.obj_info["obj_scale"] * np.ones(3)
        builder.add_visual_from_file(filename=mesh_path, scale=scale)
        builder.add_convex_collision_from_file(filename=mesh_path, scale=scale)
        self.instance = builder.build(name="object")
        self.instance.set_disable_gravity(True)
        self.instance.set_linear_damping(10)
        self.instance.set_angular_damping(10)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self.agent.reset(self.agent.keyframes["rest"].qpos)

    def _get_obs_extra(self, info: Dict):
        return dict(tcp_pose=self.agent.tcp_pose.raw_pose)

    def evaluate(self):
        false = torch.tensor([False])
        return {
            "success": false,
            "is_grasped": self.agent.is_grasping(self.instance),
        }

    def get_imagined_hand_skeleton(self):
        return torch.stack([link.pose.p for link in self.agent.all_hand_links], dim=1)

    def _process_pointcloud(self, pointcloud):
        """Crop, label (is_object, is_hand) and FPS to NUM_POINTS. Returns None if nothing is visible."""
        processed = []
        for env_id in range(pointcloud["xyzw"].shape[0]):
            points = pointcloud["xyzw"][env_id, :, :3].cpu().numpy()
            mask = ((points >= np.array([-10, -10, 0.01])) & (points <= np.array([0.5, 10, 1]))).all(axis=-1)
            points = points[mask]
            seg = pointcloud["segmentation"][env_id, mask, 0].cpu().numpy()
            seg_obj = seg >= SEG_OBJ_MIN
            onehot = np.zeros_like(points)
            onehot[seg_obj, 0] = 1
            onehot[~seg_obj, 1] = 1
            points = np.concatenate([points, onehot], axis=-1)
            if len(points) == 0:
                return None
            if len(points) < NUM_POINTS:
                pad = np.random.choice(len(points), NUM_POINTS - len(points))
                points = np.concatenate([points, points[pad]], 0)
            points = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
            sampled, _ = sample_farthest_points(points, K=NUM_POINTS)
            processed.append(sampled.squeeze(0))
        return torch.stack(processed, dim=0)

    def get_obs(self, info: Optional[Dict] = None, unflattened: bool = False):
        obs = super().get_obs(info=info, unflattened=unflattened)
        if not isinstance(obs, dict):
            return obs

        if self._obs_mode == "pointcloud":
            processed = self._process_pointcloud(obs["pointcloud"])
            if processed is None:
                return None
            obs["pointcloud"]["xyzw"] = processed

        # flatten to {state, pointcloud, hand_pcd}
        if "agent" in obs and "extra" in obs:
            obs.pop("sensor_param", None)
            obs.pop("sensor_data", None)
            data = dict(agent=obs.pop("agent"), extra=obs.pop("extra"))
            obs["state"] = common.flatten_state_dict(data, use_torch=True, device=self.device).squeeze()
            pointcloud = obs.get("pointcloud", {}).get("xyzw")
            if pointcloud is not None:
                obs["pointcloud"] = pointcloud.squeeze()
                obs["hand_pcd"] = self.get_imagined_hand_skeleton().squeeze()
            if isinstance(obs.get("pointcloud"), dict):
                obs.pop("pointcloud", None)
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(self.instance.pose.p - self.agent.tcp_pose.p, axis=1)
        return 1 - torch.tanh(5 * tcp_to_obj_dist)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
