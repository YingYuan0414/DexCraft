"""Step 5 of the MimicGen pipeline: add MimicGen `datagen_info` (eef / target / object poses, subtask
signals) to each source demo by replaying its actions in simulation.

Adapted from mimicgen/scripts/prepare_src_dataset.py. Difference: DexCraft replays the recorded actions from
the initial state instead of resetting to every recorded state.

    python scripts/mimicgen/prepare_src_dataset.py --dataset demos/SprayBottle-v1/source/merged_trajectory_mimicgen.hdf5
"""
import argparse
import json
import shutil

import h5py
import mimicgen.utils.file_utils as MG_FileUtils
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
from mimicgen.env_interfaces.base import make_interface
from tqdm import tqdm

import dexcraft.mimicgen  # noqa: F401  (registers the ManiSkill env type and MimicGen interface)


def extract_datagen_info_from_trajectory(env, env_interface, initial_state, actions):
    env.reset()
    env.reset_to(initial_state)

    all_datagen_infos = []
    for action in actions:
        all_datagen_infos.append(env_interface.get_datagen_info(action=action).to_dict())
        env.step(action)

    # list of dict -> dict of arrays (nested for object poses and subtask signals)
    all_datagen_infos = TensorUtils.list_of_flat_dict_to_dict_of_list(all_datagen_infos)
    for k in all_datagen_infos:
        if k in ["object_poses", "subtask_term_signals"]:
            all_datagen_infos[k] = TensorUtils.list_of_flat_dict_to_dict_of_list(all_datagen_infos[k])
            for k2 in all_datagen_infos[k]:
                all_datagen_infos[k][k2] = np.array(all_datagen_infos[k][k2])
        else:
            all_datagen_infos[k] = np.array(all_datagen_infos[k])
    return all_datagen_infos


def prepare_src_dataset(dataset_path, env_interface_name, env_interface_type, filter_key=None, n=None,
                        output_path=None):
    if output_path is not None:
        shutil.copy(dataset_path, output_path)
        dataset_path = output_path

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta, camera_names=[], camera_height=84, camera_width=84, reward_shaping=False
    )
    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))

    # the env interface takes the underlying simulation environment, not the robomimic wrapper
    env_interface = make_interface(name=env_interface_name, interface_type=env_interface_type, env=env.base_env)
    demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=dataset_path, filter_key=filter_key, start=None, n=n)

    print(f"File that will be modified with datagen info: {dataset_path}")
    with h5py.File(dataset_path, "a") as f:
        for ep in tqdm(demos):
            ep_grp = f[f"data/{ep}"]
            states = ep_grp["states"][()]
            actions = ep_grp["actions"][()]
            assert len(states) == len(actions)
            datagen_info = extract_datagen_info_from_trajectory(
                env=env, env_interface=env_interface, initial_state=dict(states=states[0]), actions=actions
            )

            if "datagen_info" in ep_grp:
                del ep_grp["datagen_info"]
            for k in datagen_info:
                if k in ["object_poses", "subtask_term_signals"]:
                    for k2 in datagen_info[k]:
                        ep_grp.create_dataset(f"datagen_info/{k}/{k2}", data=np.array(datagen_info[k][k2]))
                else:
                    ep_grp.create_dataset(f"datagen_info/{k}", data=np.array(datagen_info[k]))
            ep_grp["datagen_info"].attrs["env_interface_name"] = env_interface_name
            ep_grp["datagen_info"].attrs["env_interface_type"] = env_interface_type
    print(f"Modified {len(demos)} trajectories to include datagen info.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="source hdf5, modified in-place")
    parser.add_argument("--env_interface", type=str, default="MG_DexCraft")
    parser.add_argument("--env_interface_type", type=str, default="maniskill")
    parser.add_argument("--n", type=int, default=None, help="(optional) only process the first n demos")
    parser.add_argument("--filter_key", type=str, default=None, help="(optional) hdf5 filter key")
    parser.add_argument("--output", type=str, default=None, help="(optional) write to a new hdf5 instead")
    args = parser.parse_args()
    prepare_src_dataset(
        dataset_path=args.dataset,
        env_interface_name=args.env_interface,
        env_interface_type=args.env_interface_type,
        filter_key=args.filter_key,
        n=args.n,
        output_path=args.output,
    )
