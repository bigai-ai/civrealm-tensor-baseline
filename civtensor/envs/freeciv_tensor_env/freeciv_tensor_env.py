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
            "freeciv/FreecivTensor-v0", parallel_number, port_start
        )
        self.observation_spaces = self.tensor_env.observation_spaces
        self.action_spaces = self.tensor_env.action_spaces

    def reset(self):
        obs_ori, _ = self.tensor_env.reset()
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

    def step(
        self,
        actor_type,
        city_id,
        city_action_type,
        unit_id,
        unit_action_type,
        gov_action_type,
    ):
        actions = list(
            zip(
                list(actor_type),
                list(city_id),
                list(city_action_type),
                list(unit_id),
                list(unit_action_type),
                list(gov_action_type),
            )
        )
        obs_ori, rew_ori, term_ori, trun_ori, _ = self.tensor_env.step(actions)
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
        trun = np.stack(trun_ori)
        return obs, rew, term, trun
