"""Open a DexCraft task and run it with random (or zero) actions.

    python scripts/run_env.py --env-id Pen-v1 --gui
    python scripts/run_env.py --env-id Pen-v1 --obs-mode pointcloud --steps 10   # print observation shapes
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import tyro

import dexcraft.envs  # noqa: F401


@dataclass
class Args:
    env_id: str = "SprayBottle-v1"
    obs_mode: str = "state_dict"
    """'state_dict', 'pointcloud', 'none', ..."""
    control_mode: str = "pd_ee_target_delta_pose"
    gui: bool = False
    steps: int = 1000
    zero_action: bool = False
    seed: int = 0


def main(args: Args):
    env = gym.make(args.env_id, obs_mode=args.obs_mode, control_mode=args.control_mode,
                   render_mode="human" if args.gui else None)
    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    if isinstance(obs, dict):
        print({k: tuple(v.shape) for k, v in obs.items() if hasattr(v, "shape")})
    for _ in range(args.steps):
        action = np.zeros(env.action_space.shape) if args.zero_action else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if args.gui:
            env.render()
    print({k: bool(v) for k, v in info.items() if k != "elapsed_steps"})
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
