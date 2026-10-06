"""Step 2 of the MimicGen pipeline: clean up a teleoperated demo.

- drops steps with all-zero actions (keyboard idle time)
- relabels the goal to the final pose of the tool, and the target marker accordingly, so the source demo
  "reaches" its own goal

    python scripts/mimicgen/truncate_demos.py --env-id SprayBottle-v1 --demo-id 0

Reads demos/<env-id>/teleop/trajectory_<id>.{h5,json}, writes demos/<env-id>/source/trajectory_<id>.{h5,json}.
"""
import argparse
import json
from pathlib import Path

import h5py
import mani_skill.trajectory.utils as trajectory_utils
import numpy as np
from mani_skill.utils.registration import REGISTERED_ENVS
from mani_skill.utils.structs.pose import Pose

import dexcraft.envs  # noqa: F401


def write_dict(grp, d):
    for k, v in d.items():
        if isinstance(v, dict):
            write_dict(grp.create_group(k), v)
        else:
            grp.create_dataset(k, data=v)


def main(args):
    root = Path(args.record_dir) / args.env_id
    with open(root / "teleop" / f"trajectory_{args.demo_id}.json") as f:
        json_data = json.load(f)
    target_offset = REGISTERED_ENVS[args.env_id].cls.get_target_offset()

    out_dir = root / "source"
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(root / "teleop" / f"trajectory_{args.demo_id}.h5") as src, \
            h5py.File(out_dir / f"trajectory_{args.demo_id}.h5", "w") as dst:
        for episode in json_data["episodes"]:
            traj_id = f"traj_{episode['episode_id']}"
            actions = np.array(src[traj_id]["actions"])
            env_states = trajectory_utils.dict_to_list_of_dicts(src[traj_id]["env_states"])

            keep = [i for i, a in enumerate(actions) if (a != 0).any()]
            if len(keep) == 0:
                print(f"Skipping empty trajectory {traj_id}")
                continue
            env_states = trajectory_utils.list_of_dicts_to_dict([env_states[0]] + [env_states[i] for i in keep])
            print(f"{traj_id}: kept {len(keep)} / {len(actions)} steps")

            # goal := final tool pose; target marker placed relative to it
            tool_states = env_states["articulations"]["None"]
            env_states["actors"]["goal_site"][:, :7] = tool_states[-1, :7]
            goal_pose = Pose.create_from_pq(tool_states[-1, :3], tool_states[-1, 3:7])
            target_pose = goal_pose * target_offset
            env_states["actors"]["target"][:, :3] = target_pose.p.numpy()
            env_states["actors"]["target"][:, 3:7] = target_pose.q.numpy()

            grp = dst.create_group(traj_id)
            grp.create_dataset("actions", data=actions[keep])
            write_dict(grp.create_group("env_states"), env_states)

            episode["elapsed_steps"] = len(keep)
            episode["success"] = True

    with open(out_dir / f"trajectory_{args.demo_id}.json", "w") as f:
        json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--demo-id", type=int, required=True)
    parser.add_argument("--record-dir", type=str, default="demos")
    main(parser.parse_args())
