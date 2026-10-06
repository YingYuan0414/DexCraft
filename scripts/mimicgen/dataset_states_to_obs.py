"""Step 7 of the MimicGen pipeline: replay generated demos and record point-cloud observations.

Adds per-step `goal_gripper_pcd`: the hand keypoints at the end of each subtask (grasp, place, actuate),
used as goal conditioning for the policy.

    python scripts/mimicgen/dataset_states_to_obs.py --input <generated>/demo.hdf5 --output <generated>/demo_obs.hdf5 --num_workers 8
"""
import argparse
import json
import multiprocessing
from copy import deepcopy

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import torch

import dexcraft.mimicgen  # noqa: F401  (registers the ManiSkill env type)

multiprocessing.set_start_method("spawn", force=True)

SUBTASK_KEYS = ["grasp", "place"]


def obs_to_numpy(obs):
    return {k: v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v) for k, v in obs.items()}


def create_env(env_meta, args):
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta, camera_names=[], camera_height=args.camera_height, camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )


def subtask_goals(hand_pcd, subtask_signals, traj_len):
    """Hand keypoints at the end of each subtask, and per-step goal (the one of the current subtask)."""
    switch_indices = []
    for k in SUBTASK_KEYS:
        diffs = np.diff(np.array(subtask_signals[k]))
        switch_indices.append(traj_len - 1 if np.all(diffs == 0) else int(diffs.nonzero()[0][0]) + 1)
    switch_indices.append(traj_len - 1)
    switch_indices = np.maximum.accumulate(switch_indices)  # subtasks end in order

    all_goals, per_step_goals = [], []
    for index in switch_indices:
        goal = hand_pcd[index]
        all_goals.append(goal)
        n_repeat = index - len(per_step_goals) + 1
        if n_repeat > 0:
            per_step_goals.extend(np.tile(goal[None], (n_repeat, 1, 1)))
    return np.stack(all_goals, axis=0), np.stack(per_step_goals, axis=0)


def extract_trajectory(env_meta, args, initial_state, states, actions):
    env = create_env(env_meta, args)
    assert states.shape[0] == actions.shape[0]
    env.reset()
    obs = env.reset_to(initial_state)

    traj = dict(obs=[], next_obs=[], rewards=[], dones=[], actions=np.array(actions), states=np.array(states),
                initial_state_dict=initial_state)
    traj_len = states.shape[0]
    subtask_signals = []
    for t in range(1, traj_len + 1):
        info = env.env.unwrapped.evaluate()
        subtask_signals.append(dict(grasp=int(info["is_grasped"]),
                                    place=int(info["is_obj_placed"] and info["is_obj_oriented"])))

        next_obs, _, _, _ = env.step(actions[t - 1])
        done = False
        if args.done_mode in [1, 2]:  # done at the end of the trajectory
            done = done or (t == traj_len)
        if args.done_mode in [0, 2]:  # done at success states
            done = done or bool(env.is_success()["task"])

        traj["obs"].append(obs_to_numpy(obs))
        traj["next_obs"].append(obs_to_numpy(next_obs))
        traj["rewards"].append(env.get_reward().item())
        traj["dones"].append(int(done))
        obs = deepcopy(next_obs)

    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])
    subtask_signals = TensorUtils.list_of_flat_dict_to_dict_of_list(subtask_signals)

    all_goals, per_step_goals = subtask_goals(traj["obs"]["hand_pcd"], subtask_signals, traj_len)
    traj["initial_state_dict"]["goal_gripper_pcd"] = all_goals
    traj["obs"]["goal_gripper_pcd"] = per_step_goals
    all_goals_dense, _ = subtask_goals(traj["obs"]["hand_pcd_dense"], subtask_signals, traj_len)
    traj["initial_state_dict"]["goal_gripper_pcd_dense"] = all_goals_dense

    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])
    return traj


def worker(x):
    env_meta, args, initial_state, states, actions = x
    return extract_trajectory(env_meta=env_meta, args=args, initial_state=initial_state, states=states,
                              actions=actions)


def dataset_states_to_obs(args):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.input)
    env_meta["env_kwargs"]["obs_mode"] = "pointcloud"
    env = create_env(env_meta, args)
    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))

    f = h5py.File(args.input, "r")
    demos = sorted(f["data"].keys(), key=lambda d: int(d[5:]))
    if args.n is not None:
        demos = demos[: args.n]

    f_out = h5py.File(args.output, "w")
    data_grp = f_out.create_group("data")
    print(f"input file: {args.input}\noutput file: {args.output}")
    compression = None if args.no_compress else "gzip"

    total_samples = 0
    for i in range(0, len(demos), args.num_workers):
        batch = demos[i: i + args.num_workers]
        jobs = []
        for ep in batch:
            states = f[f"data/{ep}/states"][()]
            jobs.append([env_meta, args, dict(states=states[0]), states, f[f"data/{ep}/actions"][()]])
        with multiprocessing.Pool(args.num_workers) as pool:
            trajs = pool.map(worker, jobs)

        for ep, traj in zip(batch, trajs):
            # keep group names of the source file so that filter keys stay valid
            ep_grp = data_grp.create_group(ep)
            ep_grp.create_dataset("actions", data=traj["actions"])
            ep_grp.create_dataset("states/states", data=traj["states"])
            ep_grp.create_dataset("states/goal_gripper_pcd", data=traj["initial_state_dict"]["goal_gripper_pcd"])
            ep_grp.create_dataset("rewards", data=traj["rewards"])
            ep_grp.create_dataset("dones", data=traj["dones"])
            for k in traj["obs"]:
                ep_grp.create_dataset(f"obs/{k}", data=traj["obs"][k], compression=compression)
                if args.include_next_obs:
                    ep_grp.create_dataset(f"next_obs/{k}", data=traj["next_obs"][k], compression=compression)
            ep_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]
            print(f"{ep}: wrote {ep_grp.attrs['num_samples']} transitions")

    if "mask" in f:
        f.copy("mask", f_out)
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
    print(f"Wrote {len(demos)} trajectories to {args.output}")
    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="generated hdf5 (MimicGen demo.hdf5)")
    parser.add_argument("--output", type=str, required=True, help="output hdf5 with observations")
    parser.add_argument("--n", type=int, default=None, help="(optional) only process the first n demos")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--shaped", action="store_true", help="use shaped rewards")
    parser.add_argument("--camera_height", type=int, default=84)
    parser.add_argument("--camera_width", type=int, default=84)
    parser.add_argument("--done_mode", type=int, default=2,
                        help="0: done at success states, 1: done at the end of each trajectory, 2: both")
    parser.add_argument("--include_next_obs", action="store_true", help="also store next_obs")
    parser.add_argument("--no_compress", action="store_true", help="do not gzip observations")
    dataset_states_to_obs(parser.parse_args())
