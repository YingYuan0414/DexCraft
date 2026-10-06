"""Visualize a single Dexonomy grasp as a full trajectory: approach (with a side detour to avoid the object),
pregrasp -> grasp -> squeeze, then lift the object.

    python scripts/dexonomy/replay_grasp.py --dexonomy-root /path/to/DexLearn --grasp-type 1_Large_Diameter --grasp-id 0 --gui
"""
import random
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import sapien
import tyro
from mani_skill.utils.wrappers import RecordEpisode
from scipy.spatial.transform import Rotation as R

import dexcraft.envs  # noqa: F401
from dexcraft.dexonomy import LIFT_HEIGHT, list_grasp_files, load_grasp

INIT_ACTION = np.array([-0.615, 0, 0.1, np.pi / 2, 0, np.pi / 2] + [0] * 22)
STEPS_PER_SEGMENT = 120
LIFT_STEPS = 200


@dataclass
class Args:
    dexonomy_root: str
    """DexLearn repository root with the Dexonomy data"""
    grasp_type: str = "1_Large_Diameter"
    grasp_id: int = 0
    """Index into the (sorted) grasp files of this grasp type"""
    obs_mode: str = "pointcloud"
    gui: bool = False
    record_dir: Optional[str] = None
    """If set, save a video and the trajectory under <record-dir>/ShadowGrasp-v1/"""
    seed: Optional[int] = None


def interpolate_keyframes(goal_actions):
    """Joint-space keyframes from the rest pose to the squeeze pose, linearly interpolated."""
    detour = goal_actions[0].copy()
    detour[6:] = 0
    # move to the side the palm is facing away from, so the hand does not sweep through the object
    rot = R.from_euler("xyz", goal_actions[0][3:6], degrees=False).as_matrix()
    palm_y, palm_x = rot @ np.array([0, -1, 0]), rot @ np.array([-1, 0, 0])
    detour[1] += random.uniform(0.2, 0.25) * (1 if palm_y[1] < 0 else -1)
    detour[0] += random.uniform(0.1, 0.12) * (1 if palm_x[0] < 0 else -1)

    pregrasp_open = goal_actions[0].copy()
    pregrasp_open[6:] = 0
    keyframes = np.stack([INIT_ACTION, detour, pregrasp_open, *goal_actions], axis=0)
    traj = [(1 - a) * keyframes[i] + a * keyframes[i + 1]
            for i in range(len(keyframes) - 1) for a in np.linspace(0, 1, STEPS_PER_SEGMENT, endpoint=False)]
    traj.append(keyframes[-1])
    return np.array(traj)


def main(args: Args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    grasp_files = list_grasp_files(args.dexonomy_root, args.grasp_type)
    print(f"Found {len(grasp_files)} grasps for {args.grasp_type}")
    goal_actions, obj_info = load_grasp(grasp_files[args.grasp_id % len(grasp_files)], args.dexonomy_root)
    actions = interpolate_keyframes(goal_actions)

    env = gym.make("ShadowGrasp-v1", robot_uids="floating_shadow", obs_mode=args.obs_mode,
                   control_mode="pd_joint_pos", render_mode="human" if args.gui else "rgb_array",
                   enable_shadow=True, obj_info=obj_info)
    if args.record_dir:
        env = RecordEpisode(env, f"{args.record_dir}/ShadowGrasp-v1", trajectory_name=f"grasp_{args.grasp_id}",
                            save_video=True, info_on_video=False, save_trajectory=True)
    env.reset(seed=args.seed, options=dict(reconfigure=True))

    # step with deltas between consecutive keyframe interpolations
    for t in range(1, len(actions)):
        _, _, _, _, info = env.step(actions[t] - actions[t - 1])
        if args.gui:
            env.render()

    # lift: move the object up with the hand holding still, then check that it is still grasped
    obj_pose = obj_info["obj_pose"]
    for t in range(LIFT_STEPS):
        action = np.zeros(28)
        action[2] = 0.001
        env.unwrapped.instance.set_pose(
            sapien.Pose(p=[0, 0, LIFT_HEIGHT + 0.0009 * t] + obj_pose[:3], q=obj_pose[3:]))
        _, _, _, _, info = env.step(action)
        if args.gui:
            env.render()

    print("Grasp success!" if bool(info["is_grasped"]) else "Grasp failed.")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
