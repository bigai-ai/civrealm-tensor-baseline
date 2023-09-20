import os

# Disable log deduplication of Ray. This ensures the print messages from all actors can be shown.
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray

import numpy as np

from freeciv_gym.freeciv.utils.freeciv_logging import ray_logger_setup
from freeciv_gym.envs.parallel_tensor_env import ParallelTensorEnv


class FreecivTensorEnv:
    def __init__(self, parallel_number, port_start):
        ray.init(
            local_mode=False,
            runtime_env={"worker_process_setup_hook": ray_logger_setup},
        )
        self.logger = ray_logger_setup()
        self.tensor_env = ParallelTensorEnv(
            "freeciv/FreecivTensorMinitask-v0", parallel_number, port_start
        )
        self.observation_spaces = self.tensor_env.observation_spaces
        self.action_spaces = self.tensor_env.action_spaces

    def reset(self):
        obs_ori, _ = self.tensor_env.reset(minitask_pattern="buildcity")
        obs = {}
        for key in [
            "rules",
            "map",
            "player",
            "city",
            "unit",
            "others_player",
            "others_unit",
            "others_city",
            "unit_mask",
            "city_mask",
            "others_unit_mask",
            "others_city_mask",
            "others_player_mask",
            "actor_type_mask",
            "city_id_mask",
            "city_action_type_mask",
            "unit_id_mask",
            "unit_action_type_mask",
            "gov_action_type_mask",
        ]:
            obs[key] = np.stack([obs_single[key] for obs_single in obs_ori])
        return obs

    def step(self, actions_ori):
        keys = [
            "actor_type",
            "city_id",
            "city_action_type",
            "unit_id",
            "unit_action_type",
            "gov_action_type",
        ]
        batch_size = actions_ori["actor_type"].shape[0]
        actions = []
        for i in range(batch_size):
            actions.append({key: actions_ori[key][i] for key in keys})
        obs_ori, rew_ori, term_ori, trunc_ori, _ = self.tensor_env.step(actions)
        obs = {}
        for key in [
            "rules",
            "map",
            "player",
            "city",
            "unit",
            "others_player",
            "others_unit",
            "others_city",
            "unit_mask",
            "city_mask",
            "others_unit_mask",
            "others_city_mask",
            "others_player_mask",
            "actor_type_mask",
            "city_id_mask",
            "city_action_type_mask",
            "unit_id_mask",
            "unit_action_type_mask",
            "gov_action_type_mask",
        ]:
            obs[key] = np.stack([obs_single[key] for obs_single in obs_ori])
        rew = np.stack(rew_ori)
        term = np.stack(term_ori)
        trunc = np.stack(trunc_ori)
        return (
            obs,
            np.expand_dims(rew, -1),
            np.expand_dims(term, -1),
            np.expand_dims(trunc, -1),
        )
