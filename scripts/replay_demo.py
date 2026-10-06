"""Replay a recorded ManiSkill trajectory (teleop or source demo), save a video and report success.

    python scripts/replay_demo.py --traj-path demos/SprayBottle-v1/source/trajectory_0.h5
"""
import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.trajectory.utils as trajectory_utils
import numpy as np
import tyro
from mani_skill.utils.wrappers import RecordEpisode

import dexcraft.envs  # noqa: F401


@dataclass
class Args:
    traj_path: str
    """ManiSkill trajectory file (.h5, with a .json next to it)"""
    gui: bool = False
    """Show the GUI instead of only writing a video"""
    save_video: bool = True
    """Save the replay as <traj-name>.mp4 next to the trajectory"""


def main(args: Args):
    traj_path = Path(args.traj_path)
    with open(traj_path.with_suffix(".json")) as f:
        json_data = json.load(f)
    env_kwargs = dict(json_data["env_info"]["env_kwargs"])
    env_kwargs["render_mode"] = "human" if args.gui else "rgb_array"
    env = gym.make(json_data["env_info"]["env_id"], **env_kwargs)
    if args.save_video:
        env = RecordEpisode(env, str(traj_path.parent), video_fps=30, save_trajectory=False, info_on_video=False)

    with h5py.File(traj_path) as trajectory_data:
        for episode in json_data["episodes"]:
            data = trajectory_data[f"traj_{episode['episode_id']}"]
            env.reset(**episode["reset_kwargs"])
            env.unwrapped.set_state_dict(trajectory_utils.dict_to_list_of_dicts(data["env_states"])[0])
            first_success = None
            for t, action in enumerate(np.array(data["actions"])):
                _, _, _, _, info = env.step(action)
                if first_success is None and bool(info["success"]):
                    first_success = t
                if args.gui:
                    env.render()
            print(f"episode {episode['episode_id']}: {len(data['actions'])} steps, first success at step {first_success}, "
                  f"success at the end: {bool(info['success'])}")
            if args.save_video:
                env.flush_video(name=f"{traj_path.stem}_ep{episode['episode_id']}")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
