"""Collect a demonstration with keyboard teleoperation.

    python scripts/teleop.py --env-id SprayBottle-v1 --demo-id 0

Saves demos/<env-id>/teleop/trajectory_<demo-id>.{h5,json,mp4}. Press `q` to finish.
"""
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import sapien
import tyro
from mani_skill.utils.wrappers import RecordEpisode

import dexcraft.envs  # noqa: F401  (registers envs)
from dexcraft.teleop import TELEOP_CONTROL_MODE, all_keys, get_pressed_keys, get_teleop_action, keymap_help


@dataclass
class Args:
    env_id: str = "SprayBottle-v1"
    """DexCraft task to teleoperate"""
    record_dir: Optional[str] = "demos"
    """Root directory for recordings (set to None to not record)"""
    demo_id: int = 0
    """Index of the saved trajectory"""
    seed: Optional[int] = None
    """Seed for the initial object / goal poses"""
    shader: str = "default"
    """Shader for rendering: 'default', 'rt' or 'rt-fast'"""
    save_video: bool = True
    verbose: bool = False


def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)
    env = gym.make(
        args.env_id,
        robot_uids="xarm6_leap",
        obs_mode="none",
        control_mode=TELEOP_CONTROL_MODE,
        render_mode="human",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        enable_shadow=True,
    )
    if args.record_dir:
        record_dir = f"{args.record_dir}/{args.env_id}/teleop"
        env = RecordEpisode(env, record_dir, trajectory_name=f"trajectory_{args.demo_id}", save_video=args.save_video,
                            info_on_video=False, save_trajectory=True, source_type="teleoperation")

    env.reset(seed=args.seed, options=dict(reconfigure=True))
    viewer = env.render()
    assert isinstance(viewer, sapien.utils.Viewer)
    print(keymap_help(args.env_id))

    keys = all_keys(args.env_id)
    while True:
        pressed_keys = get_pressed_keys(viewer, keys)
        if "q" in pressed_keys:
            break
        action = get_teleop_action(pressed_keys, args.env_id)
        _, _, _, _, info = env.step(action)
        if args.verbose:
            print({k: bool(v) for k, v in info.items() if k.startswith("is_") or k == "success"})
        env.render()
    env.close()
    if args.record_dir:
        print(f"Saved trajectory to {record_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
