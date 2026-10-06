"""Turn Dexonomy grasps into hand-object trajectories for pre-training.

For each grasp: motion-plan the floating Shadow hand from a random pose to the pregrasp pose (avoiding the
object point cloud), close the fingers through pregrasp -> grasp -> squeeze, then move to a random pose.
Trajectories in which the hand never grasps the object are discarded.

    python scripts/dexonomy/collect_dataset.py --dexonomy-root /path/to/DexLearn --grasp-type 1_Large_Diameter \
        --output data/dexonomy/1_Large_Diameter --n 100 --num-workers 16

Each trajectory is saved as <output>/demo_<i>/{actions, state, pointcloud, hand_pcd, goal_gripper_pcd}.npy
"""
import argparse
import collections
import json
import multiprocessing
import random
import time
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import transforms3d.quaternions as tq
from mplib import Planner, Pose
from scipy.spatial.transform import Rotation as R

import dexcraft.envs  # noqa: F401
from dexcraft.agents import FloatingShadow
from dexcraft.dexonomy import list_grasp_files, load_grasp

multiprocessing.set_start_method("spawn", force=True)

REST_ACTION = np.array([-0.615, 0, 0.1, np.pi / 2, 0, np.pi / 2] + [0] * 22)
GRASP_OFFSET_RANGE = (0.2, 0.3, 0.1)
FINGER_STEPS = 100  # steps to interpolate between keyframes / finger configurations


def list_of_dict_to_dict_of_list(list_of_dict):
    dic = collections.OrderedDict()
    for d in list_of_dict:
        for k, v in d.items():
            dic.setdefault(k, []).append(v.numpy())
    return dic


def make_keyframes(goal_actions):
    """Random start -> pregrasp (fingers open) -> pregrasp -> grasp -> squeeze -> random end (holding)."""
    start = np.concatenate([
        [random.uniform(-0.6, 0.2), random.uniform(-0.35, 0.35), random.uniform(0.1, 0.3),
         random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)],
        np.zeros(22),
    ])
    end = np.concatenate([
        [random.uniform(-0.6, 0.2), random.uniform(-0.35, 0.35), random.uniform(0.3, 0.5)],
        goal_actions[2][3:],
    ])
    pregrasp_open = goal_actions[0].copy()
    pregrasp_open[6:] = 0
    if random.random() < 0.1:
        start = REST_ACTION.copy()
    return np.stack([start, pregrasp_open, goal_actions[0], goal_actions[1], goal_actions[2], end], axis=0)


def extract_trajectory(env_kwargs, env_id, grasp_path, dexonomy_root):
    goal_actions, obj_info = load_grasp(grasp_path, dexonomy_root, offset_range=GRASP_OFFSET_RANGE)
    keyframes = make_keyframes(goal_actions)

    env = gym.make(env_id, **env_kwargs, obj_info=obj_info)
    obs, _ = env.reset(seed=None, options=dict(reconfigure=True))

    traj = dict(obs=[], dones=[], actions=[], states=[])
    traj_len = 800
    valid_grasp = False
    planner = Planner(urdf=FloatingShadow.urdf_path, srdf=FloatingShadow.srdf_path, move_group="palm")

    switch_indices = []  # end of the pregrasp, grasp and squeeze stages
    for t in range(len(keyframes)):
        state = env.unwrapped.get_state()
        obj_pts = obs["pointcloud"][np.where(obs["pointcloud"][:, 3] == 1)[0]][:, :3]
        planner.update_point_cloud(obj_pts)  # avoid the object when planning
        keyframe = keyframes[t]
        if t <= 2:
            # plan the wrist motion to the first keyframes, fingers open
            quat = tq.mat2quat(R.from_euler("xyz", keyframe[3:6], degrees=False).as_matrix())
            qpos = env.unwrapped.agent.robot.get_qpos().numpy().reshape(-1)
            result = planner.plan_screw(Pose(p=keyframe[:3], q=quat), qpos, time_step=0.01)
            if result["status"] != "Success":
                result = planner.plan_pose(Pose(p=keyframe[:3], q=quat), qpos, time_step=0.01)
            if "position" not in result:
                return None
            n_step = result["position"].shape[0]
            interp_traj = np.concatenate([result["position"], np.zeros((n_step, 22))], axis=-1)
            interp_traj[:, [3, 5]] = interp_traj[:, [5, 3]]  # planner and controller order rotation joints differently
        else:
            interp_traj = np.array([(1 - a) * keyframes[t - 1] + a * keyframe
                                    for a in np.linspace(0, 1, FINGER_STEPS, endpoint=True)])
        if len(interp_traj) == 0:
            continue

        # interpolate the fingers separately, so they move over at least FINGER_STEPS steps
        last_fingers = np.zeros(22) if t == 0 else keyframes[t - 1][6:]
        target_fingers = keyframe[6:]
        total_len = max(FINGER_STEPS, len(interp_traj)) if target_fingers.sum() != 0 else len(interp_traj)
        finger_traj = np.array([(1 - a) * last_fingers + a * target_fingers
                                for a in np.linspace(0, 1, total_len, endpoint=True)])
        interp_traj = np.concatenate([interp_traj, np.tile(interp_traj[-1], (total_len - len(interp_traj), 1))], axis=0)
        interp_traj[:, 6:] = finger_traj

        rel_actions = interp_traj[1:] - interp_traj[:-1]
        for at, a in enumerate(rel_actions):
            next_obs, _, _, _, info = env.step(a)
            done = False
            if info["is_grasped"]:
                valid_grasp = True
                traj_len = len(traj["obs"]) + 1
            if t == len(keyframes) - 1 and at == len(rel_actions) - 1 and info["is_grasped"]:
                done = True
                traj_len = len(traj["obs"]) + 1

            if t > 0:  # skip the approach from the random start pose
                if at == len(rel_actions) - 1 and t in [1, 4, 5]:
                    switch_indices.append(len(traj["obs"]))
                traj["obs"].append(obs)
                traj["dones"].append(int(done))
                traj["actions"].append(a)
                traj["states"].append(state.squeeze().numpy())
            obs = deepcopy(next_obs)

    print("Valid grasp:", valid_grasp)
    if not valid_grasp or any(o is None for o in traj["obs"]):
        return None
    # truncate after the last grasped step
    for i in range(len(switch_indices)):
        if traj_len - 1 <= switch_indices[i]:
            switch_indices = switch_indices[:i] + [traj_len - 1]
            break
    traj["states"].append(env.unwrapped.get_state().squeeze().numpy())
    traj["obs"] = list_of_dict_to_dict_of_list(traj["obs"])

    # goal hand keypoints of the current stage, for every step
    goal_gripper_pcd = []
    for index in switch_indices:
        goal = traj["obs"]["hand_pcd"][index]
        goal_gripper_pcd.extend(np.tile(goal[None], (index - len(goal_gripper_pcd) + 1, 1, 1)))
    traj["obs"]["goal_gripper_pcd"] = np.array(goal_gripper_pcd)

    for k in traj:
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp][:traj_len])
        else:
            traj[k] = np.array(traj[k][:traj_len])
    return traj


def worker(x):
    return extract_trajectory(*x)


def main(args):
    grasp_files = list_grasp_files(args.dexonomy_root, args.grasp_type)
    print(f"Found {len(grasp_files)} grasps for {args.grasp_type}")

    env_kwargs = dict(
        robot_uids="floating_shadow",
        obs_mode="pointcloud",
        control_mode="pd_joint_target_delta_pos",
        render_mode="rgb_array",
        enable_shadow=True,
    )
    print(json.dumps(dict(env_name=args.env_id, env_kwargs=env_kwargs), indent=4))

    start = time.time()
    n_valid = 0
    for i in range(0, len(grasp_files), args.num_workers):
        if args.n is not None and n_valid >= args.n:
            break
        batch = grasp_files[i: i + args.num_workers]
        with multiprocessing.Pool(args.num_workers) as pool:
            trajs = pool.map(worker, [(env_kwargs, args.env_id, g, args.dexonomy_root) for g in batch])

        for j, traj in enumerate(trajs):
            if args.n is not None and n_valid >= args.n:
                break
            if traj is None:
                print(f"grasp {i + j} is invalid, skipped")
                continue
            out_dir = Path(args.output) / f"demo_{n_valid}"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "actions.npy", np.asarray(traj["actions"], dtype=np.float32))
            for key in ["state", "goal_gripper_pcd", "hand_pcd", "pointcloud"]:
                np.save(out_dir / f"{key}.npy", np.asarray(traj["obs"][key], dtype=np.float32))
            n_valid += 1
            print(f"{i + j + 1}/{len(grasp_files)}: valid {n_valid}, saved to {out_dir} ({time.time() - start:.1f}s)")
    print(f"Saved {n_valid} trajectories in {time.time() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dexonomy-root", type=str, required=True, help="DexLearn repository with the Dexonomy data")
    parser.add_argument("--grasp-type", type=str, required=True, help="e.g. 1_Large_Diameter")
    parser.add_argument("--output", type=str, required=True, help="output directory")
    parser.add_argument("--env-id", type=str, default="ShadowGrasp-v1")
    parser.add_argument("--n", type=int, default=None, help="(optional) stop after n valid trajectories")
    parser.add_argument("--num-workers", type=int, default=2)
    main(parser.parse_args())
