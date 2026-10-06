"""Shared implementation of the DexCraft tool-use tasks.

Every task follows the same structure: an articulated tool (from PartNet-Mobility) lies on the table,
and the robot (xArm6 + LEAP hand) must grasp it, move it to a goal pose (shown as a green ghost), and
actuate its joint (e.g. press the spray-bottle trigger). Task subclasses only set the class attributes
below and, optionally, add extra scene actors via `_load_extra_actors` / `_initialize_extra_actors`.
"""
import json
import pickle
from typing import Any, Dict, Optional

import numpy as np
import sapien
import torch
from mani_skill.agents.registration import REGISTERED_AGENTS
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from pytorch3d.ops import sample_farthest_points
from scipy.spatial.transform import Rotation as R

import dexcraft.agents  # noqa: F401  (registers robots)
from dexcraft.utils import ASSET_DIR, generate_random_quat, quaternion_distance

# Point cloud observation settings
NUM_POINTS = 1024
SEG_HAND_MIN, SEG_OBJ_MIN = 8, 26  # segmentation ids: [8, 26) are hand links, >= 26 are task objects


def build_target_marker(scene: ManiSkillScene, radius=0.04, depth=0.001):
    """Visual-only square marker showing where the tool should point when actuated."""
    builder = scene.create_actor_builder()
    half_thickness = radius * 0.5
    half_sizes = [
        [depth, half_thickness, radius],
        [depth, half_thickness, radius],
        [depth, radius, half_thickness],
        [depth, radius, half_thickness],
    ]
    poses = [
        sapien.Pose([0, half_thickness, 0]),
        sapien.Pose([0, -half_thickness, 0]),
        sapien.Pose([0, 0, half_thickness]),
        sapien.Pose([0, 0, -half_thickness]),
    ]
    mat = sapien.render.RenderMaterial(base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5)
    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_visual(pose, half_size, material=mat)
    return builder


class DexCraftEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["xarm6_leap"]

    # --- robot ---
    agent_disable_self_collisions = True

    # --- tool (PartNet-Mobility id under assets/sapien) ---
    obj_id: int
    obj_scale: float
    obj_half_length = 0.08  # spawn height of the tool
    obj_friction = (5.0, 0.8)  # (static, dynamic)
    obj_joint = dict(joint="hinge")  # entry in mobility_v2.json of the joint to actuate
    obj_joint_drive = (1.0, 0.5)  # (stiffness, damping)
    obj_joint_drive_target = 0.2
    obj_rot_lock = (True, True)  # (lock_x, lock_y) of the initial tool orientation
    obj_rot_bounds = ((0, np.pi / 3), (-np.pi / 3, 0), (5 * np.pi / 6, 7 * np.pi / 6))  # extrinsic xyz euler
    step_after_load_tool = False  # run one physics step right after loading the tool (kept for reproducibility)

    # --- goal ---
    goal_rot_bounds = ((0, np.pi / 3), (-np.pi / 3, 0), (5 * np.pi / 6, 7 * np.pi / 6))
    goal_min_height = 0.1  # goal is at least this much higher than the initial tool
    target_offset = ([-0.08, 0, 0.12], ("y", 15))  # target marker pose relative to goal: (xyz, (euler axis, deg))

    # --- success ---
    goal_thresh = 0.04  # m
    orient_thresh = 30  # deg
    trigger_thresh = (">=", 0.3)  # tool joint qpos condition for "actuated"

    # --- workspace / cameras ---
    spawn_half_size = 0.05
    spawn_center = (-0.1, 0)
    max_goal_height = 0.3
    sensor_cam_eye_pos = [[0.3, 0, 0.6], [0, 0.3, 0.6]]
    sensor_cam_target_pos = [-0.1, 0, 0.1]
    human_cam_eye_pos = [0.6, 0.7, 0.6]
    human_cam_target_pos = [0.0, 0.0, 0.35]

    def __init__(self, *args, robot_uids="xarm6_leap", robot_init_qpos_noise=0.02, **kwargs):
        if robot_uids not in self.SUPPORTED_ROBOTS:
            raise NotImplementedError(f"{robot_uids} is not supported, choose from {self.SUPPORTED_ROBOTS}")
        self.robot_init_qpos_noise = robot_init_qpos_noise
        with open(ASSET_DIR / "robot" / "leap_hand" / "hand_dense.pkl", "rb") as f:
            self.dense_hand_points = pickle.load(f)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @classmethod
    def get_target_offset(cls):
        pos, (axis, deg) = cls.target_offset
        return Pose.create_from_pq(pos, R.from_euler(axis, deg, degrees=True).as_quat())

    # ------------------------------------------------------------------ #
    # Cameras & agent
    # ------------------------------------------------------------------ #
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
        # self-collision of the robot is a per-task setting, applied while the agent is built
        agent_cls = REGISTERED_AGENTS[self.robot_uids].agent_cls
        default = agent_cls.disable_self_collisions
        agent_cls.disable_self_collisions = self.agent_disable_self_collisions
        try:
            super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))
        finally:
            agent_cls.disable_self_collisions = default

    # ------------------------------------------------------------------ #
    # Scene
    # ------------------------------------------------------------------ #
    def _load_tool(self):
        obj_dir = ASSET_DIR / "sapien" / str(self.obj_id)
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = True
        loader.fix_root_link = False
        loader.set_material(static_friction=self.obj_friction[0], dynamic_friction=self.obj_friction[1], restitution=0.0)
        loader.scale = self.obj_scale
        loader.set_density(1000)
        instance = loader.load(str(obj_dir / "mobility.urdf"))

        # index of the actuated joint among the active joints (parsed from the PartNet-Mobility annotation)
        with open(obj_dir / "mobility_v2.json") as f:
            joint_entries = json.load(f)
        dof, joint_index = 0, None
        for entry in joint_entries:
            if entry["joint"] == "free":
                dof += 1
            if all(entry[k] == v for k, v in self.obj_joint.items()):
                joint_index = dof - 1
        assert (dof == instance.dof).all(), f"dof parse error for {self.obj_id}: {dof} vs {instance.dof}"
        assert joint_index is not None, f"no joint matching {self.obj_joint} in {self.obj_id}"
        return instance, joint_index

    def _load_goal(self):
        builder = self.scene.create_actor_builder()
        builder.set_initial_pose(sapien.Pose())
        mesh_path = str(ASSET_DIR / "sapien" / str(self.obj_id) / "combined.obj")
        builder.add_visual_from_file(filename=mesh_path, scale=self.obj_scale * np.ones(3), material=[0, 1, 0])
        return builder.build_kinematic(name="goal_site")

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.instance, self.joint_index = self._load_tool()
        self.instance.get_active_joints()[0].set_drive_properties(
            stiffness=self.obj_joint_drive[0], damping=self.obj_joint_drive[1]
        )
        if self.step_after_load_tool:
            self.scene.step()

        self.goal_site = self._load_goal()
        builder = build_target_marker(self.scene)
        builder.set_initial_pose(sapien.Pose([0, 0, 0.01]))
        self.target = builder.build_kinematic(name="target")
        self._load_extra_actors()
        self._hidden_objects.append(self.goal_site)

    def _load_extra_actors(self):
        """Hook for task-specific actors (e.g. a stand)."""

    def _initialize_extra_actors(self, obj_xyz, obj_quat):
        """Hook to place task-specific actors given the initial tool pose."""

    def _sample_xy(self, b):
        xy = torch.rand((b, 2)) * self.spawn_half_size * 2 - self.spawn_half_size
        xy[:, 0] += self.spawn_center[0]
        xy[:, 1] += self.spawn_center[1]
        return xy

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.instance.get_active_joints()[0].set_drive_target(self.obj_joint_drive_target)

        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.agent.reset(self.agent.keyframes["rest"].qpos)

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = self._sample_xy(b)
            xyz[:, 2] = self.obj_half_length
            lock_x, lock_y = self.obj_rot_lock
            qs = generate_random_quat(b, lock_x=lock_x, lock_y=lock_y, bounds=self.obj_rot_bounds)
            self.instance.set_pose(Pose.create_from_pq(xyz, qs))
            self._initialize_extra_actors(xyz, qs)

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = self._sample_xy(b)
            goal_xyz[:, 2] = torch.rand((b)) * (self.max_goal_height - 0.1) + xyz[:, 2] + self.goal_min_height
            goal_qs = generate_random_quat(b, lock_x=True, bounds=self.goal_rot_bounds)
            goal_pose = Pose.create_from_pq(goal_xyz, goal_qs)
            self.goal_site.set_pose(goal_pose)
            self.target.set_pose(goal_pose * self.get_target_offset())

    # ------------------------------------------------------------------ #
    # Evaluation & reward
    # ------------------------------------------------------------------ #
    def evaluate(self):
        is_obj_placed = torch.linalg.norm(self.goal_site.pose.p - self.instance.pose.p, axis=1) <= self.goal_thresh
        is_obj_oriented = (
            quaternion_distance(self.goal_site.pose.q, self.instance.pose.q) * 180 / np.pi <= self.orient_thresh
        )
        op, thresh = self.trigger_thresh
        joint_qpos = self.instance.get_qpos()[..., self.joint_index]
        is_handle_triggered = joint_qpos >= thresh if op == ">=" else joint_qpos <= thresh
        return {
            "success": is_obj_placed & is_obj_oriented & is_handle_triggered,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": self.agent.is_static(0.2),
            "is_obj_oriented": is_obj_oriented,
            "is_handle_triggered": is_handle_triggered,
            "is_grasped": self.agent.is_grasping(self.instance),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(self.instance.pose.p - self.agent.tcp_pose.p, axis=1)
        reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        obj_to_goal_dist = torch.linalg.norm(self.goal_site.pose.p - self.instance.pose.p, axis=1)
        reward += 1 - torch.tanh(5 * obj_to_goal_dist)
        obj_to_goal_orient = quaternion_distance(self.goal_site.pose.q, self.instance.pose.q)
        reward += 1 - torch.tanh(5 * obj_to_goal_orient)

        reward[info["success"]] = 5
        reward[info["is_obj_placed"]] = 2
        reward[info["is_obj_oriented"]] = 2
        reward[info["is_handle_triggered"]] = 1
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def _get_obs_extra(self, info: Dict):
        return dict(tcp_pose=self.agent.tcp_pose.raw_pose)

    def get_imagined_hand_skeleton(self):
        """Positions of the hand links (palm, finger joints and tips), [B, 16, 3]."""
        return torch.stack([link.pose.p for link in self.agent.all_hand_links], dim=1)

    def get_imagined_hand_dense(self):
        """Hand point cloud from points pre-sampled on each LEAP link mesh, [1, 136, 3]."""
        art = self.agent.robot._objs[0]
        art_points = []
        for link in art.links:
            if "world" in link.name or "link" in link.name or "root" in link.name:
                continue
            T = self.agent.robot.links[link.index].pose[0].sp.to_transformation_matrix()
            local_pts = self.dense_hand_points[link.name.split("leap_")[-1]]
            art_points.append((T[:3, :3] @ local_pts.T).T + T[:3, 3])
        art_points = np.concatenate(art_points, axis=0)
        return torch.tensor(art_points).unsqueeze(0).to(self.device)

    def _process_pointcloud(self, pointcloud):
        """Crop to the workspace, keep hand + object points with a one-hot label, FPS to NUM_POINTS.

        Returns [B, NUM_POINTS, 6] (xyz, is_object, is_hand, 0).
        """
        processed = []
        for env_id in range(pointcloud["xyzw"].shape[0]):
            points = pointcloud["xyzw"][env_id, :, :3].cpu().numpy()
            mask = ((points >= np.array([-10, -10, 0.01])) & (points <= np.array([0.5, 10, 1]))).all(axis=-1)
            points = points[mask]

            seg = pointcloud["segmentation"][env_id, mask, 0].cpu().numpy()
            seg_obj = seg >= SEG_OBJ_MIN
            seg_hand = (seg < SEG_OBJ_MIN) & (seg >= SEG_HAND_MIN)
            keep = seg_obj | seg_hand  # drops table and arm points
            points = points[keep]
            onehot = np.zeros_like(points)
            onehot[seg_obj[keep], 0] = 1
            onehot[seg_hand[keep], 1] = 1
            points = np.concatenate([points, onehot], axis=-1)

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
            obs["pointcloud"]["xyzw"] = self._process_pointcloud(obs["pointcloud"])

        # flatten to {state, pointcloud, rgb, hand_pcd, hand_pcd_dense}
        if "agent" in obs and "extra" in obs:
            obs.pop("sensor_param", None)
            obs.pop("sensor_data", None)
            data = dict(agent=obs.pop("agent"), extra=obs.pop("extra"))
            obs["state"] = common.flatten_state_dict(data, use_torch=True, device=self.device).squeeze()

            pointcloud = obs.get("pointcloud", {}).get("xyzw")
            if pointcloud is not None:
                obs["rgb"] = obs["pointcloud"]["rgb"].reshape(-1, 128, 128, 3).cpu().numpy()
                obs["pointcloud"] = pointcloud.squeeze()
                obs["hand_pcd"] = self.get_imagined_hand_skeleton().squeeze()
                obs["hand_pcd_dense"] = self.get_imagined_hand_dense().squeeze()
            if isinstance(obs.get("pointcloud"), dict):
                obs.pop("pointcloud", None)
        return obs
