"""Step 4 of the MimicGen pipeline: convert a merged ManiSkill trajectory file into robomimic's hdf5 format
(flattened env states + actions, 80/20 train/valid split).

    python scripts/mimicgen/convert_to_robomimic.py --dataset demos/SprayBottle-v1/source/merged_trajectory.h5

Writes merged_trajectory_mimicgen.hdf5 next to the input.
"""
import argparse
import json

import gymnasium as gym
import h5py
import mani_skill.trajectory.utils as trajectory_utils
import numpy as np
from robomimic.scripts.split_train_val import split_train_val_from_hdf5

import dexcraft.envs  # noqa: F401
from dexcraft.mimicgen.env_wrapper import MANISKILL_TYPE


def state_layout(env_id, env_kwargs):
    """(group, name) pairs in the order ManiSkill flattens the env state."""
    env = gym.make(env_id, robot_uids=env_kwargs.get("robot_uids", "xarm6_leap"), obs_mode="none")
    state_dict = env.unwrapped.get_state_dict()
    env.close()
    return [(group, name) for group in state_dict for name in state_dict[group]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="merged ManiSkill trajectory (.h5)")
    args = parser.parse_args()

    h5_path = args.dataset
    output_path = h5_path.replace(".h5", "_mimicgen.hdf5")
    with open(h5_path.replace(".h5", ".json")) as f:
        json_data = json.load(f)

    env_id = json_data["env_info"]["env_id"]
    env_kwargs = json_data["env_info"]["env_kwargs"]
    env_kwargs["render_mode"] = "rgb_array"
    env_meta = dict(type=MANISKILL_TYPE, env_name=env_id, env_kwargs=env_kwargs)
    layout = state_layout(env_id, env_kwargs)

    with h5py.File(h5_path) as trajectory_data, h5py.File(output_path, "w") as f:
        data_group = f.create_group("data")
        data_group.attrs["env_args"] = json.dumps(env_meta, indent=4)
        total_samples = 0
        for episode in json_data["episodes"]:
            data = trajectory_data[f"traj_{episode['episode_id']}"]
            env_states = trajectory_utils.dict_to_list_of_dicts(data["env_states"])
            states = np.stack([np.concatenate([s[group][name] for group, name in layout]) for s in env_states])

            demo = data_group.create_group(f"demo_{episode['episode_id']}")
            demo.attrs["num_samples"] = len(data["actions"])
            demo.create_dataset("states", data=states[:-1])  # one state per action
            demo.create_dataset("actions", data=data["actions"][:])
            total_samples += len(data["actions"])
        data_group.attrs["total"] = total_samples

    split_train_val_from_hdf5(hdf5_path=output_path, val_ratio=0.2)
    print(f"Wrote {output_path}")
