"""Loading Dexonomy grasps (https://github.com/JYChen18/DexLearn) for the floating Shadow hand.

Expected layout under `dexonomy_root` (the DexLearn repository with the Dexonomy data downloaded):
    assets/grasp/Dexonomy_GRASP_shadow/succ_collect/<grasp_type>/<object>/floating/scaleXXX.npy
    assets/object/...   (scene configs and meshes referenced by the grasp files)
"""
import os
import random
from glob import glob

import numpy as np
import transforms3d.quaternions as tq
from scipy.spatial.transform import Rotation as R

GRASP_SUBDIR = "assets/grasp/Dexonomy_GRASP_shadow/succ_collect"
GRASP_STAGES = ["pregrasp", "grasp", "squeeze"]
LIFT_HEIGHT = 0.2  # grasps and objects are lifted by this much above the table


def list_grasp_types(dexonomy_root):
    return sorted(os.listdir(os.path.join(dexonomy_root, GRASP_SUBDIR)))


def list_grasp_files(dexonomy_root, grasp_type):
    grasp_dir = os.path.join(dexonomy_root, GRASP_SUBDIR, grasp_type)
    return sorted(glob(os.path.join(grasp_dir, "**/*.npy"), recursive=True))


def load_grasp(grasp_path, dexonomy_root, offset_range=(0.3, 0.3, 0.1)):
    """Load a Dexonomy grasp and its object, randomly translated by up to `offset_range` (x, y, z).

    Returns:
        goal_actions: [3, 28] hand configurations (xyz, extrinsic xyz euler, 22 joints) for pregrasp, grasp, squeeze
        obj_info: dict(obj_path, obj_scale, obj_pose) for ShadowGrasp-v1
    """
    grasp_data = np.load(grasp_path, allow_pickle=True).item()
    rand_offset = np.array([random.uniform(-r, r) for r in offset_range])

    goal_actions = []
    for stage in GRASP_STAGES:
        qpos = grasp_data[f"{stage}_qpos"]
        qpos = qpos[0] if len(qpos.shape) > 1 else qpos  # only use the first grasp
        rot = R.from_matrix(tq.quat2mat(qpos[3:7])).as_euler("xyz", degrees=False)
        action = np.concatenate([qpos[:3], rot, qpos[7:]])
        action[2] += LIFT_HEIGHT
        goal_actions.append(action)
    goal_actions = np.array(goal_actions)
    goal_actions[:, :3] += rand_offset[None, :]

    scene_path = os.path.join(dexonomy_root, grasp_data["scene_path"])
    scene_cfg = np.load(scene_path, allow_pickle=True).item()
    obj_name = scene_cfg["task"]["obj_name"]
    obj_cfg = scene_cfg["scene"][obj_name]
    obj_pose = obj_cfg["pose"]
    obj_pose[:3] += rand_offset.tolist()
    obj_path = os.path.join(scene_path.split("scale")[0], os.path.dirname(os.path.dirname(obj_cfg["file_path"])))
    obj_info = dict(obj_path=obj_path, obj_scale=obj_cfg["scale"][0] * grasp_data["scene_scale"][0], obj_pose=obj_pose)
    return goal_actions, obj_info
