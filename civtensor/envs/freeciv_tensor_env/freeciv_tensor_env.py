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
        obs, _ = self.tensor_env.reset()
        return_list = []
        for key in [
            "rules",
            "player",
            "other_players",
            "units",
            "cities",
            "other_units",
            "other_cities",
            "civmap",
            "other_players_mask",
            "units_mask",
            "cities_mask",
            "other_units_mask",
            "other_cities_mask",
            "actor_type_mask",
            "city_id_mask",
            "city_action_type_mask",
            "unit_id_mask",
            "unit_action_type_mask",
            "gov_action_type_mask",
        ]:
            return_list.append(np.stack([obs_single[key] for obs_single in obs]))
        return return_list

    def step(
        self,
        actor_type,
        city_id,
        city_action_type,
        unit_id,
        unit_action_type,
        gov_action_type,
    ):
        actions = zip(
            list(actor_type),
            list(city_id),
            list(city_action_type),
            list(unit_id),
            list(unit_action_type),
            list(gov_action_type),
        )
        obs, rew, term, trun, _ = self.tensor_env.step(actions)
        return_list = []
        for key in [
            "rules",
            "player",
            "other_players",
            "units",
            "cities",
            "other_units",
            "other_cities",
            "civmap",
            "other_players_mask",
            "units_mask",
            "cities_mask",
            "other_units_mask",
            "other_cities_mask",
            "actor_type_mask",
            "city_id_mask",
            "city_action_type_mask",
            "unit_id_mask",
            "unit_action_type_mask",
            "gov_action_type_mask",
        ]:
            return_list.append(np.stack(obs_single[key] for obs_single in obs))
        return_list.append(np.stack(rew))
        return_list.append(np.stack(term))
        return_list.append(np.stack(trun))
        return return_list
