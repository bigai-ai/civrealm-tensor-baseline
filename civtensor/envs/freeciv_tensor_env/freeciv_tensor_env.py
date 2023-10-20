import os

# Disable log deduplication of Ray. This ensures the print messages from all actors can be shown.
os.environ["RAY_DEDUP_LOGS"] = "0"
import numpy as np
import ray
from civrealm.envs.freeciv_minitask_env import MinitaskType
from civrealm.envs.parallel_tensor_env import ParallelTensorEnv
from civrealm.freeciv.utils.freeciv_logging import ray_logger_setup


class TensorBaselineEnv:
    def __init__(self, parallel_number, port_start, task="fullgame"):
        ray.init(
            local_mode=False,
            runtime_env={"worker_process_setup_hook": ray_logger_setup},
        )
        self.logger = ray_logger_setup()
        self.task_args = task.split(" ")
        task_type = self.task_args[0]
        if task_type == "fullgame":
            self.tensor_env = ParallelTensorEnv(
                "freeciv/FreecivTensor-v0", parallel_number, port_start
            )
        elif task_type in MinitaskType.list() or task_type == "random_minitask":
            task = {} if task_type == "random_minitask" else {"type": task_type}
            if len(self.task_args) > 1:
                assert self.task_args[1] in ['easy', 'normal', 'hard']
                task["level"] = self.task_args[1]
            if len(self.task_args) > 2:
                task["id"] = self.task_args[2]
            self.tensor_env = ParallelTensorEnv(
                "freeciv/FreecivTensorMinitask-v0",
                parallel_number,
                port_start,
                minitask_pattern=task,
            )
        else:
            raise ValueError(
                f"Expected task type in {['fullgame','random_minitask']+MinitaskType.list()} but got {task}"
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
        obs_ori, rew_ori, term_ori, trunc_ori, info_ori = self.tensor_env.step(actions)
        # print(self.tensor_env.get_recent_scores())
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
        score_keys = set(sum((list(info["scores"].keys()) for info in info_ori), []))
        scores = {
            k: np.array(
                [info["scores"][k] if k in info["scores"] else 0 for info in info_ori]
            )
            for k in score_keys
        }
        return (
            obs,
            np.expand_dims(rew, -1),
            np.expand_dims(term, -1),
            np.expand_dims(trunc, -1),
            scores,
        )

    def close(self):
        try:
            self.tensor_env.close()
        except Exception as e:
            print(e)
        ray.shutdown()
