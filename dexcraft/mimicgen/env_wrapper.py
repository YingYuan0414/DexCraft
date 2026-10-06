"""robomimic EnvBase wrapper around ManiSkill3 environments (needed by MimicGen)."""
from copy import deepcopy

import gymnasium as gym
import numpy as np
import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils
import torch
from mani_skill.utils import common

import dexcraft.envs  # noqa: F401  (registers envs)

MANISKILL_TYPE = 4  # robomimic EnvType id used in dataset metadata (`env_args["type"]`)


class EnvManiSkill(EB.EnvBase):
    def __init__(self, env_name: str, robot_uids: str = "xarm6_leap", **kwargs):
        self._init_kwargs = deepcopy(kwargs)
        self._env_name = env_name
        self._current_obs = None
        self._current_reward = None
        self._current_done = None
        self._current_info = None
        self.env = gym.make(env_name, robot_uids=robot_uids, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_obs = obs
        self._current_reward = reward
        self._current_done = terminated or truncated
        self._current_info = info
        return obs, reward, self._current_done, info

    def reset(self):
        # seed from numpy's global RNG (seeded by MimicGen from the config), so generation is reproducible
        self.env.reset(seed=int(np.random.randint(2**31)), options=dict(reconfigure=True))
        self._current_obs = None
        self._current_reward = None
        self._current_done = None
        self._current_info = None

    def reset_to(self, state):
        self.env.reset(options=dict(reconfigure=True))
        if len(state["states"].shape) == 1:
            state["states"] = np.expand_dims(state["states"], axis=0)
        assert len(state["states"].shape) == 2, "State should be a 2D array"
        self.env.unwrapped.set_state(state["states"])
        return self.get_observation()

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        if mode == "human":
            return self.env.render_human()
        if mode == "rgb_array":
            return self.env.render_rgb_array(camera_name=camera_name)
        raise NotImplementedError(f"mode={mode} is not implemented")

    def get_observation(self, obs=None):
        if obs is None:
            obs = self.env.get_obs()
            self._current_obs = obs
        return obs

    def is_success(self):
        if self._current_info is None:
            return {"task": False}
        return {"task": self._current_info["success"]}

    def serialize(self):
        return {"type": MANISKILL_TYPE, "env_name": self._env_name, "env_kwargs": deepcopy(self._init_kwargs)}

    def get_goal(self):
        return self.env.unwrapped.goal_site.pose.raw_pose

    def set_goal(self, goal):
        with torch.device(self.env.unwrapped.device):
            self.env.unwrapped.goal_site.set_pose(goal)

    def get_state(self):
        state = common.flatten_state_dict(self.env.unwrapped.get_state_dict(), use_torch=True)
        return {"states": state}

    @classmethod
    def create_for_data_processing(cls, env_name, camera_names, camera_height, camera_width, reward_shaping,
                                   render=None, render_offscreen=None, use_image_obs=None, use_depth_obs=None,
                                   **kwargs):
        ObsUtils.initialize_obs_utils_with_obs_specs(
            {"obs": {"low_dim": ["state"], "rgb": ["pointcloud", "hand_pcd"]}}
        )
        return cls(env_name=env_name, **kwargs)

    @property
    def action_dimension(self):
        return self.env.action_space.shape[0]

    @property
    def base_env(self):
        return self.env

    def get_reward(self):
        return self._current_reward

    def is_done(self):
        return self._current_done

    @property
    def name(self):
        return self._env_name

    @property
    def rollout_exceptions(self):
        return tuple()

    @property
    def type(self):
        return MANISKILL_TYPE
